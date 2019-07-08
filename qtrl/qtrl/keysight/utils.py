# Copyright (c) 2018-2019, UC Regents

from qtrl.keysight import keysightSD1 as sd1
from collections import namedtuple

KeysightSlot = namedtuple('KeysightSlot', ['channels', 'chassis', 'slot', 'model', 'type', 'connection'])
KeysightChannel = namedtuple('KeysightChannel', ['channel', 'chassis', 'slot', 'model', 'type', 'connection'])

ks_errors = {x: y for y, x in dict(sd1.SD_Error.__dict__).items() if isinstance(x, int)}


class KeysightInfo():
    card_types = {1: 'HVI',
                  2: 'DAC',
                  3: 'TDC',
                  4: 'DIO',
                  5: 'WAVE',
                  6: 'ADC',
                  7: "AIO"}
    
    card_channels = {'M3202A': 4,
                     'M3102A': 4}


def get_cards():
    """Returns a list of cards, each card is returned as a
    tuple containing (chassis number, slot, model)"""
    
    num_slots = sd1.SD_Module.moduleCount()
    models = [sd1.SD_Module.getProductNameByIndex(s) for s in range(num_slots)]
    slots = [sd1.SD_Module.getSlotByIndex(s) for s in range(num_slots)]
    types = [KeysightInfo.card_types[sd1.SD_Module.getTypeByIndex(s)] for s in range(num_slots)]
    chassis = [sd1.SD_Module.getChassisByIndex(s) for s in range(num_slots)]
    channels = [KeysightInfo.card_channels[model] for model in models]

    cards = sorted(zip(channels, chassis, slots, models, types, len(slots)*[None]), key=lambda z: z[1])

    return [KeysightSlot(*x) for x in cards]
