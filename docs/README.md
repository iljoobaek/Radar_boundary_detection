## A Python Parser for TI mmWave AWR1443 Data Packet ##

This is a fork from https://github.com/m6c7l/pymmw

So to display with pymmw, the data is placed in dictionary with suitable key.

This is part of a project in RTML.


Compared to pymmw, some modifications are as follows:
- /DATA: contains the parser for data packet from mmWave. Currently support azimuth-only heat map data packets.
- /pymmw/app/plot.py: add support of the parser into original codebase.
- /pymmw/app/plot_range_azimuth_heat_map.py: flip the np array to fit the corrent orientation.


----

The parts written by slimcatarch are under MIT license, owned by slimcatarch and RTML. 

The rest of the code and infrastructure is owned by m6c7l.

Copyright 2019 Slimcatarch & RTML

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
