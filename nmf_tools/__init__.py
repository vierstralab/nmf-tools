###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

__title__ = 'nmf-tools'
__description__ = ''
__url__ = 'https://github.com/vierstralab/nmf-tools.git'
__version__ = '1.0.0'
__author__ = 'Alexandr Boytsov'
__author_email__ = 'sboytsov@altius.org'
__license__ = 'GPL3'

import os
from matplotlib import style
import functools

def in_vierstra_style(func):
    '''
    Decorator to apply common style to a function that generates a plot.
    Style is defined in vierstragroup_matplotlibrc.
    '''
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        style_file_path = os.path.join(os.path.dirname(__file__), 'vierstragroup_matplotlibrc')

        with style.context(style_file_path):
            return func(*args, **kwargs)
    return wrapper
