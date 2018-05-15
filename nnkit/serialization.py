# The MIT License (MIT)
#
# Copyright (c) 2018 Federico Saldarini
# https://www.linkedin.com/in/federicosaldarini
# https://github.com/saldavonschwartz
# http://0xfede.io
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from . import *
import json as jsn
import gzip


def save(net, path):
    """Save a net topology (list of tuples) to a gziped json file."""
    json = []

    for n in net.topology:
        json.append({
            'op': n[0].__name__,
            'args': [
                (n.data.tolist() if type(n) is NetVar else n)
                for n in n[1:]
            ]
        })

    with gzip.open(path + '.model.gz', 'wb') as file:
        bytes = bytearray(jsn.dumps(json), encoding='utf-8')
        file.write(bytes)


def load(path):
    """Load net topology (list of tuples) from a gziped json file."""
    with gzip.open(path + '.model.gz', 'rb') as file:
        json = jsn.load(file)
        topology = [(
            eval(d['op']),
            *[
                (NetVar(v) if type(v) is list else v)
                for v in d['args']
            ]
        )

            for d in json
        ]

        return topology
