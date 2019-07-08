'''
script to control a minicircuits variable attenuator via HTTP requests. it is suggested to use the minicircuits GUI to find the ip_address.

example usage:
#open the device
qattn = m.MiniCircuitsAttenuator(ip_address='172.16.0.20')
#set the attenuation
qattn.attenuation(5)
#read the attenuation
qattn.attenuation()


#some references:
https://www.minicircuits.com/pdfs/RCDAT-6000-30.pdf
http://www.minicircuits.com/softwaredownload/Prog_Manual-6-Programmable_Attenuator.pdf
#ethernet control is page 137 in programming manual

'''

import urllib.request as urllib2
import socket


class MiniCircuitsAttenuator():

    def __init__(self, serial_number=None, ip_address=None):
        self.ip_address = None
        if ip_address is not None:
            self.ip_address = ip_address
        elif serial_number is not None:
            attenuators = find_attenuators()
            for att in attenuators:
                if att['Serial Number'] == serial_number:
                    self.ip_address = att['IP Address'][0]
        else:
            raise Exception("No serial number or IP address specified")

        if not self.validate_ip(self.ip_address):
            raise ValueError('Not a valid IP address')
    
    def snapshot(self):
        """
        Returns a snapshot of the MiniCircuitsAttenuator
        :return: dictionary of address, model, s/n, and attenuation
        """
        snapshot = {'parameters': {'address': self.ip_address,
                    'model': self.model(),
                    'serial_number': self.serial_number(),
                    'attenuation': self.attenuation()}
                    }
        return snapshot
    def write(self, command=None):
        """gets status of source or turns on/off the source"""
        ip_address = self.ip_address
        string = 'http://{:s}/{:s}'.format(ip_address, command)
        response = urllib2.urlopen(string)
        return response.read()

    def attenuation(self, attn=None):
        """gets status of source or turns on/off the source"""
        if attn is None:
            attn = self.write('{:s}'.format(':ATT?'))
            return float(attn)# * dBm
        else:
            status = self.write('{:s}={:f}'.format(':SETATT', attn))
            if status is '2':
                raise ValueError('Set attenuation failed. Out of range')
            elif status is '0':
                raise ValueError('Invalid attenuation setting')

    def model(self):
        """gets status of source or turns on/off the source"""

        mn = self.write('{:s}'.format(':MN?'))
        return mn

    def serial_number(self):
        """gets status of source or turns on/off the source"""

        sn = self.write('{:s}'.format(':SN?'))
        return sn

    def validate_ip(self, s):
        a = s.split('.')
        if len(a) != 4:
            return False
        for x in a:
            if not x.isdigit():
                return False
            i = int(x)
            if i < 0 or i > 255:
                return False
        return True


def find_attenuators():
    cs = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cs.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    cs.sendto('MCLDAT?', ('255.255.255.255', 4950))
    cs.settimeout(1)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(('', 4951))
    s.settimeout(.1)

    attenuators = []

    for i in range(5):
        try:
            m = s.recvfrom(1024)
            attenuators.append(parse_attenuator_msg(m[0]))
        except:
            break

    return attenuators


def parse_attenuator_msg(message):
    attenuator_msg = message.split('\r\n')[0:-1]

    attenuator = {}
    for msg in attenuator_msg:
        if 'Serial Number:' in msg:
            attenuator['Serial Number'] = msg.strip('Serial Number: ')
        elif 'IP Address=' in msg:
            attenuator['IP Address'] = msg.strip('IP Address=').split('  Port: ')
        elif 'Model Name:' in msg:
            attenuator['Model Name'] = msg.strip('Model Name: ')

    return attenuator
