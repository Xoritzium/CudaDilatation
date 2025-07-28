from DilatationGPU import DilatationGPU
from DilatationCPU import DilatationCPU
import argparse

class Main:

    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-m", "--mask", type=int, help ="provide the radius of the mask, it will be treated as m*m matrix")
        parser.add_argument("-g", action="store_true", help ="run the calculation of the dilataion on GPU")
        parser.add_argument("-c", action="store_true", help ="run the calculation of the dilataion on CPU")

        args = parser.parse_args()  
        if args.g:
            print("starting to apply the dilatation on gpu")
            d_gpu = DilatationGPU()
            # d_gpu.run_calculation(args.mask)
        elif args.c:
            print("starting to apply the dilatation on cpu")
            d_cpu= DilatationCPU()
            d_cpu.run_calculation(args.mask)  
        else:
            print("no argument chosen - check help")
            return

if __name__=="__main__":
    main = Main()
    main.main()