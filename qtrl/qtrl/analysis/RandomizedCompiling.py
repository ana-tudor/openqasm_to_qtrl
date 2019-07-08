from __future__ import print_function
from .EasyGates import *
from itertools import product
import copy
from collections import namedtuple

Twirl = namedtuple('Twirl', ['input_twirl', 'output_twirl'])


def coherent_noisy_rotation(noise, n_qubits=2):
    rotations = map(lambda x: tensor(*x), product([sigmax, sigmay, sigmaz, np.eye(2)], repeat=n_qubits))
    rotations = np.random.randn(len(rotations))[:, np.newaxis, np.newaxis]*rotations
    rotations = np.sum(rotations, 0)*noise
    return np.matrix(expm(1j/2*(rotations)))


def find_min_set(gateset, max_depth=3, preferred_gates=['Z', 'I']):
    """Take a gateset and find the minimum number of elements which
    close the set. Additionally returns a dictionary which maps all possible
    min_step + 1 combinations down to -> 1 arbitrary min_step equivalency"""
    
    keys = list(gateset.keys())
    gateset_vals = np.array(list(gateset.values()))
    states = copy.copy(gateset_vals)

    qudit = gateset_vals.shape[-1]
    
    depth = 0
    n_steps = len(states)
    for i in range(max_depth):
        depth += 1
        print("Trying to close set with {} operation(s).".format(depth))
        old_copy = copy.deepcopy(states).conj().reshape(-1, qudit, qudit)
        states = np.einsum('...abc,dce->...adbe', states, gateset_vals)
            
        undo = np.einsum('...abc,dec->...adbe', states, old_copy)
        
        undo = remove_phase(undo)
        states = remove_phase(states)
        
        if np.max(is_ident(undo), -1).all():
            print("Set is closed in {} operations".format(depth))
            
            reshape_list = (2*depth+1)*[n_steps]
            reshape_list.extend([qudit, qudit])
            undo_ind = np.argwhere(is_ident(undo.reshape(*reshape_list)))
            reductions = {}
            for c in np.take(keys, undo_ind, axis=0).tolist():
                if isinstance(c[0], list):
                    gatelist = [gate for sublist in c[:depth+1] for gate in sublist]
                    replacement = [gate for sublist in c[depth+1:] for gate in sublist]
                else:
                    gatelist = c[:depth+1]
                    replacement = c[depth+1:]

                if tuple(gatelist) not in reductions:
                    reductions[tuple(gatelist)] = replacement

            return depth, reductions
        
    raise Exception("No closure found in depth of:", depth)

def build_id_simplifications(commuting_gates, identity='I'):
    simplifications = {}
    for gate in commuting_gates:
        if gate != identity:
            simplifications[(gate, identity)] = (identity, gate)
        
    return simplifications


def twirl_rep(twirl, hard=['CNOT', 'NULL']):
    gates = map(list, zip(*twirl.input_twirl))
    gates.append(hard)
    gates.extend(map(list, zip(*twirl.output_twirl)))
    
    return gates


def sort_reductions(simplifications, identity='I'):
    new_simplifications = {}
    for key in simplifications:
        new_key = []
        for entry in key:
            if entry != identity:
                new_key.append(entry)
                
        new_val = []
        for entry in simplifications[key]:
            if entry != identity:
                new_val.append(entry)
                
        # the replacement has to be the same length, pad with Identity
        while len(new_val) < len(new_key):
            new_val.insert(0, identity)
        
        if len(new_key) > 0 and new_key != new_val:
            new_simplifications[tuple(new_key)] = tuple(new_val)
        
    return new_simplifications

def equivalencies(gateset, reps, save=False, name=None):
    
    if save and name is None:
        raise Exception("Provide a name")
    
    hashes = {}
    for combo in product(gateset.keys(), repeat=reps):
        calc = reduce(np.dot, map(gateset.__getitem__, combo))
        calc = np.array(calc, dtype='complex64')        
        calc = np.around(calc, 3) + 0
        inds = np.array(np.nonzero(calc))[:,0]
        calc *= np.exp(-1j*np.angle(calc[inds[0], inds[1]]))
        calc = np.around(calc, 5) + 0
        if calc.tostring() in hashes:
            hashes[calc.tostring()].append(combo)
        else:
            hashes[calc.tostring()] = [combo]
        
    if save:
        print("Saving equivalencies as {}".format(name))
        with open(name, 'wb') as f:
            pickle.dump(hashes, f)
    return list(hashes.values())
    

def decompose_into_gates(U, gateset=pauli_set, return_all=False):
    """Decomposes a unitary into what gates will construct the unitary
        Accept:
            U - a unitary matrix
            gateset - pauli_set - a dicionary of allowed gates
        Returns:
            (Gate names, error in reconstruction, global phase offset)
    """
    n_qubits = int(U.shape[0]**0.5)

    gate_combos = {}
    for combo in product(list(gateset.keys()), repeat=n_qubits):
        guess = tensor(*[gateset[x] for x in combo])

        # If my guess is correct, it should undo the unitary U up to global phase
        undo_test = guess.H * U

        # we need to calculate the global phase
        phase = np.angle(undo_test[0,0])

        #now we measure the distance from I
        error = np.linalg.norm(np.exp(1j*phase)*np.eye(2**n_qubits) - undo_test)

        if np.round(error, 3) not in gate_combos:
            gate_combos[np.round(error, 3)] = [[combo, phase]]
        else:
            gate_combos[np.round(error, 3)].append([combo, phase])

    best_error = np.min(list(gate_combos.keys()))
    if return_all:
        return gate_combos[best_error], best_error
    else:
        return gate_combos[best_error][0], best_error


def Id_up_to_phase(U):
    phase = np.angle(U[...,0,0])
    dim = np.shape(U)[...,0]
    if np.linalg.norm(np.exp(1j*phase)*np.eye(dim) - U) < 1e-3:
        return True
    else:
        return False
    

def is_ident(array):
    n_dims = np.shape(array)[-1]
    array = remove_phase(array)
    return np.isclose(np.linalg.norm(array - np.eye(n_dims), axis=(-2, -1)), 0)
def remove_phase(array):
    array = np.array(array)

    # record our shape
    old_shape = array.shape

    # Take the first row of the matrix and vectorize all other indexes
    reshaped = array[..., 0, :].reshape(-1, old_shape[-1])

    # find the indexs of the first values which are non-zero
    non_zero_index = np.argmax(~np.isclose(np.abs(reshaped), 0), axis=1)

    # calculate the phase of the first non-zero entries
    phase = np.angle(np.array([reshaped[i, j] for i, j in enumerate(non_zero_index)])).reshape(old_shape[:-2])

    # Remove the phase
    array = array * np.exp(-1j * phase[..., np.newaxis, np.newaxis])

    return array
# def remove_phase(array):
#
#     phase = np.angle(array[..., 0, 0]+0)
#     where_zero = np.where(np.around(array[..., :, 0], 5) == 0)
#
#     if len(array.shape) > 2:
#         phase[where_zero] = np.array(np.angle(array[..., 1, 0]+0))[where_zero]
#     elif len(where_zero[0]) == 1:
#         phase = np.angle(array[1, 0]+0)
#
#     array *= np.exp(-1j*phase)[..., np.newaxis, np.newaxis]
#     return array


def find_allowed_twirls(easy_gateset=pauli_set, hard_gate=cnot):
    working_twirls = []
    for twirl in product(easy_gateset.keys(), repeat=2):
        twirl_unitary = tensor(easy_gateset[twirl[0]], easy_gateset[twirl[1]])

        undo_unitary = hard_gate*twirl_unitary.H*hard_gate.H

        decomposed_undo, error = decompose_into_gates(undo_unitary,
                                                      gateset=easy_gateset,
                                                      return_all=True)
        if error == 0:
            for output_twirl, phase in decomposed_undo:
                working_twirls.append(Twirl(twirl, output_twirl))

    return working_twirls



def decompose_unitary(U, gateset, depth=3, return_full_tree=False, allow_repeats=False, verbosity=0):
    """Constructs a tree of possible gate combinations up to the specified depth,
    and returns the series of allowed operations which undoes the provided unitary.
        Accepts:
            U - a unitary matrix you wish to decompose
            gateset - a dictionary of gates which are allowed operations
            depth - how deep the tree should go to find results, use carefully as
                    this scales horrendously.
            return_full_tree - returns the full tree which was constructed
            allow_repeats - nodes of the tree are killed when a repeat unitary is constructed

        Returns:
            dictionary with keys being the depth of the tree and values being a list of operations
            The dictionary will return empty if no decomposition found.

            if return_full_tree is true, then the complete dictionary is returned, including
            all unitaries calculated at each node of the tree."""
    U = np.matrix(U)

    gate_tree = {"Tree" : {"Node" : [],
                           'U' : U.H,
                           'Name' : []},
                 "Hashes" : set([hash(U.H.tostring())])}

    def new_node(new_U, name):
        return {"Node": [],
                'U': new_U,
                'Name': name}

    undo_operations = {}

    current_layer = [gate_tree['Tree']]
    try:
        operations_completed = 0
        pruned_nodes = 0
        for d in range(depth):
            if verbosity > 0:
                print("Working on layer {}".format(d + 1))
            next_layer = []
            if verbosity > 1:
                operations_completed += len(current_layer)
                print("   This layer has {} nodes ({:0.0f}% of the tree).".format(len(current_layer), 100.*float(len(current_layer))/len(gate_tree['Hashes'])))
                print("   There are currently {} nodes in the tree".format(len(gate_tree['Hashes'])))
                print("   There have been {} computed operations".format(operations_completed))
                print("   There have been {} pruned nodes".format(pruned_nodes))
            for node in current_layer:
                new_undos = []
                for gate in gateset:
                    new_U = node['U'] * gateset[gate]
                    # we count the unitary as the same if there is just a
                    # global phase difference, so we calculate the hash
                    # after a undoing the phase of the first element

                    # adding 0 makes -0 become +0, weird but cool
                    new_U = np.exp(-1j*np.angle(new_U[0,0])) * new_U + 0.0

                    # name should be a list of all previous nodes
                    new_name = copy.copy(node['Name'])
                    new_name.append(gate)

                    if Id_up_to_phase(new_U):
                        new_undos.append(new_name)
                        if d+1 not in undo_operations:
                            undo_operations[d+1] = [new_name]
                        else:
                            undo_operations[d+1].append(new_name)
                        continue
                    elif not allow_repeats:
                        hash_U = hash(np.round(new_U, 5).tostring())
                        if hash_U in gate_tree['Hashes']:
                            pruned_nodes += 1
                            continue
                            print("Shouldn't get here")
                        gate_tree['Hashes'].add(hash_U)

                    node_temp = new_node(new_U, new_name)

                    # So we dont have empty nodes on return, just for cleanliness
                    if d == depth - 1:
                        del node_temp['Node']

                    next_layer.append(node_temp)
                    node['Node'].append(node_temp)

                del node['U']
                if verbosity > 0 and new_undos != []:
                    print('Found {} undo operations in layer {}'.format(len(undo_operations[d+1]), d+1))

            # break out early if we can
            if next_layer == []:
                break

            current_layer = next_layer
    except KeyboardInterrupt:
        pass

    if return_full_tree:
        return undo_operations, gate_tree['Tree']
    else:
        return undo_operations


def dumb_decomp(U, gateset, depth=3, stop_on_first=False, randomize=False):

    n_ops = len(gateset)**depth
    # print("Performing {:,} Operations".format(n_ops))

    init_product = np.matrix(U).H

    if randomize:
        print("Randomizing all operations...", end='')
        combo_list = list(product(gateset, repeat=depth))
        np.random.shuffle(combo_list)
        print("Done.\nBeginning Sampling...")
    else:
        combo_list = product(gateset, repeat=depth)

    result = set()
    cur_op = 0
    for combo in combo_list:
        cur_op += 1
        if cur_op % 1e6 == 0:
            print("Progress {:.1f}%".format(cur_op/float(n_ops)*100))
        if depth != 1:
            U_product = init_product * np.linalg.multi_dot(map(gateset.__getitem__, combo))
        else:
            U_product = np.dot(init_product, map(gateset.__getitem__, combo))

        if Id_up_to_phase(U_product):
            result.add(combo)
            if stop_on_first:
                break
    return result


def decompose_unitary_new(U, gateset, depth=3, verbosity=0, force_recompute=False, stop_after_n=None):
    """ create a product generator of the gateset repeated 1 to depth times

        for each of these times, find the combo-1 entry in the previously calculated
        combo_to_hash_lookup -> hash_to_product_lookup
        use this to calculate the new product and its hash

        put the hash in the lookup table if it is not already there

        add a combo-to-hash lookup to the dictionary
    """
    init_product = np.matrix(U).H

    combo_to_hash_lookup = {}
    hash_to_product_lookup = {}
    ignore_combos = set()
    undo_operations = {}
    n_combos = 0

    for d in range(depth):
        print("Working on gate depth {}".format(d+1))
        for combo in product(gateset, repeat=(d + 1)):
            n_combos += 1

            if d == 0:
                U_product = np.round(init_product * gateset[combo[-1]], 6)
            else:
                if combo[:-1] in ignore_combos:
                    ignore_combos.add(combo)
                    continue
                if combo[:-1] not in combo_to_hash_lookup:
                    if verbosity > 1:
                        print("Couldnt find {}, recalculating it".format(combo[:-1]))
                    U_product = np.round(init_product * reduce(np.dot, map(gateset.__getitem__, combo)), 6)
                else:
                    old_hash = combo_to_hash_lookup[combo[:-1]]
                    U_product = np.round(hash_to_product_lookup[old_hash] * gateset[combo[-1]], 6)

            # remove global phase offset and adding 0 makes -0 -> 0 so hashing
            # will be consistent
            U_product = np.exp(-1j*np.angle(U_product[0,0])) * U_product + 0.0

            if Id_up_to_phase(U_product):
                if d+1 not in undo_operations:
                    undo_operations[d+1] = [combo]
                else:
                    undo_operations[d+1].append(combo)
                if stop_after_n is not None and len(undo_operations.values()) > stop_after_n:
                    break
                    break
            # continue
            p_hash = hash(U_product.tostring())
            if p_hash in hash_to_product_lookup:
                ignore_combos.add(combo)
            elif not force_recompute:
                hash_to_product_lookup[p_hash] = U_product
                combo_to_hash_lookup[combo] = p_hash

    print("Calculated {} combos, {} unique hashes".format(n_combos, len(hash_to_product_lookup)))
    return undo_operations


