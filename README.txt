openqasm_compiler.py may be used as follows:

Include both openqasm_compiler.py and qtrl_repr.py in the same folder.
To translate an OpenQASM file, run
py -m openqasm_compiler path/to/file_name.qasm

The compiler will save a new file named file_name_qtrl.txt with a list of gates
in the qtrl input format.
