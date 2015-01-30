# Copyright (c) 2014 eBay Software Foundation
# Licensed under the MIT License
#
# Generate the c++ source files used for the convolution unit tests.
# This code generation approach is not strictly necessary in this project,
# but is useful for when testing kernels that need template instantiation.
import ConfigParser
import argparse
import os
from math import ceil

def isMaxwell():
    return int(os.getenv('maxdnn_maxwell', '0')) != 0

def writeHeaderFile(config, headerfile):
    out = open(headerfile, 'w')
    out.write(getIncludeHeaders())
    for section in config.sections():
        out.write('\n')
        out.write(getConvDecl(config, section))

def writeSourceFile(config, srcfile, headerfile):
    out = open(srcfile, 'w')
    out.write(getHeaders(headerfile))
    
def writeTestFile(config, testfile, headerfile):
    out = open(testfile, 'w')
    out.write(getTestHeaders(headerfile))
    out.write(getTestFixtures(config))
    out.write(getSuite(config))

def getTestFixtures(config):
    code = ''
    for section in config.sections():
        if isMaxwell():
            code = code + getTestFixture_maxwell_64(config, section)
        else:
            raise Exception("This release require a Maxwell GPU.")
    return code

def getSuite(config):
    code = '''SUITE(convolution)
{
'''
    for section in config.sections():
        code = code + getTest(section)
    code = code + '''}
'''
    return code

def getTest(section):
    return '''TEST_FIXTURE(Convolution_%s, convolve_maxdnn_%s)
{
    testConvolution(&Convolution_%s::_convolve_gpu);
}

TEST_FIXTURE(Convolution_%s, convolve_cudnn_%s)
{
    testConvolution_cuDNN("convolve:%s:_convolve_cudnn");
}

''' % (section, section, section, section, section, section)

def getTestFixture_maxwell_64(config, section):
    # input shape
    Ni = config.getint(section, 'Ni')
    Hi = config.getint(section, 'Hi')
    Wi = config.getint(section, 'Wi')
    Nb = config.getint(section, 'Nb')
    padding_i = config.getint(section, 'padding_i')

    # convolution parameters
    Sk = config.getint(section, 'Sk')
    stride = config.getint(section, 'stride')
    padding = config.getint(section, 'padding')

    # output shape
    No = config.getint(section, 'No')
    if config.has_option(section, 'Ho'):
        Ho = config.getint(section, 'Ho')
    else:
        Ho = 1 + int(ceil((2*padding + Hi - Sk)/float(stride)))
    if config.has_option(section, 'Wo'):
        Wo = config.getint(section, 'Wo')
    else:
        Wo = 1 + int(ceil((2*padding + Wi - Sk)/float(stride)))


    return '''struct Convolution_%s : public ConvolutionFixture
{
    Convolution_%s()
    {
        Padding = %d;
        Stride = %d;
        KernelSize = %d;
        NumFilters = %d;
        BatchSize = %d;
        NumColors = %d;
        W_in = %d;
        H_in = %d;
        W_out = %d;
        H_out = %d;
        ResultFileName = getTestFile("convolve_%s.erd");
    }

    static void _convolve_gpu(const Tensor<Float> &inputs, const Tensor<Float> &filters, Tensor<Float> &output)
    {
      MAXDNN_SCOPE("convolve:%s:multiconvolution_64_maxwell");
      multiconvolution_64(inputs, filters, output, %d, %d, %f);
    }
};
''' % (section, section,
       padding,
       stride,
       Sk,
       No,
       Nb,
       Ni,
       Wi,
       Hi,
       Wo,
       Ho,
       section,
       section,
       stride,
       padding,
       1.0)

def getTestHeaders(headerfile):
    return '''#include "maxdnn/%s"
#include "maxdnn/profile.hpp"
#include "maxdnn/multiconvolution_64.hpp"
''' % headerfile

def getIncludeHeaders():
    return '''#include "maxdnn/Texture.hpp"
#include "maxdnn/Tensor.hpp"'''

def getConvDecl(config, section):
    return """
void convolution_%s(maxdnn::Texture &images_texture,
                             const maxdnn::Shape& images_shape,
                             maxdnn::Texture &filters_texture,
                             maxdnn::Tensor<float> &maps);""" % (section)

def getHeaders(headerfile):
    return '''#include "maxdnn/%s"
#include "maxdnn/ConvolutionIndexesGpu.hpp"
__constant__ maxdnn::ConvolutionIndexesGpu c_Indexes;
#include "maxdnn/ConvolutionIndexes.hpp"
#include "maxdnn/multiconvolution_64.hpp"
#include "maxdnn/profile.hpp"
'''  % (headerfile)

def main():
    parser = argparse.ArgumentParser(prog='ConvBuilder', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('configfile', help='A convolution configuration file.')
    parser.add_argument('srcfile', help='output CUDA source file')
    parser.add_argument('headerfile', help='output CUDA header file')
    parser.add_argument('testfile', help='output UnitTest++ test file')
    args = parser.parse_args()
    config = ConfigParser.ConfigParser()
    config.read(args.configfile)
    writeSourceFile(config, args.srcfile, args.headerfile)
    writeHeaderFile(config, args.headerfile)
    writeTestFile(config, args.testfile, args.headerfile)

if __name__ == "__main__":
    main()
