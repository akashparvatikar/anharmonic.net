# File setup_c.py
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('',parent_package,top_path)

    config.add_library(name='normalize_c', sources=['normalize_c.cxx'])
    config.add_extension('_norm_c',
                         sources = ['normalize_c.pyf','normalize_c_wrap.c'],
                         libraries = ['normalize_c'])
    return config
if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
