#Switch board to switch between versions
from versions.v000 import v000
from versions.v010 import v010
from versions.v011 import v011
from versions.v020 import v020
from versions.v021 import v021
from versions.v030 import v030
from versions.v031 import v031 

tested_versions = ['000','010','011','020','021','022','030','031']
unfinished_versions = ['011','021'] #the versions i need to go back to and add parameterization

version_name = "Switch Board"

default_file = 'sourceimages/rowtest.png' #tested with julians images
default_parameter_list = []

def switchBoard(version_to_run,src_image,parameter_list):
    match version_to_run:
        case '000':
            print("Running v000 - Raw")
            v000.run(src_image,parameter_list)
        case '010':
            print("Running v010 - Canny")
            v010.run(src_image,parameter_list)
        case '011':
            print("Running v011 - Canny+")
            v011.run(src_image,parameter_list)
        case '020':
            print("Running v020 - GreenExtract")
            v020.run(src_image,parameter_list)
        case '021':
            print("Running v021 - Improved Variable Extract")
            v021.run(src_image,parameter_list)
        case '030':
            print("Running v030 - Skeletonization")
            v030.run(src_image,parameter_list)
        case '031':
            print("Running v031 - Skeletonization (alternate)")
            v031.run(src_image,parameter_list)
        
        #v040 and v050 may be rolled into a default function like linedrawer
        case '040':
            print("Running v040 - Window")
            v040.run(src_image,parameter_list)
        case '050':
            print("Running v050 - Overlapping Window")
            v050.run(src_image,parameter_list)

        #2 star rating difficuilty
        case '060':
            print("Running v060 - Morphological Processing")
            v060.run(src_image,parameter_list)

        #alpha release candidate
        case '100':
            print("Running Alpha v100 - All Combined")
            v100.run(src_image,parameter_list)

#modules to check
#HE CE CQ GE KM LD MO SK

def runall():
    print("Running all",version_name,"Test Bench")
    for x in tested_versions:
        if x not in unfinished_versions:
            switchBoard(x,default_file,default_parameter_list)

def testbench():
    print("Running",version_name,"Test Bench")
    switchBoard('020',default_file,default_parameter_list)

if __name__ == "__main__":
    code = input('Enter function\n')
    if code == 'all':
        runall()
    else:
        switchBoard(str(code),default_file,default_parameter_list)