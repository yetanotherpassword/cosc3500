Instruction to build on getafix are incorporated as comments at the start of ann_mnist_digits.cpp
And repeated here:

     unzip Project_AC.zip
     cd ~/cosc3500/
     unzip mnist.zip
     unxz armadillo-10.6.2.tar.xz
     tar xvf armadillo-10.6.2.tar
     cd armadillo-10.6.2/
     mkdir build
     cd build
     cmake ..
     make
     cd ../..
     make
     sbatch ./goslurm.sh ann_mnist_digits_cuda    #Run parallel version (with default settings)
     sbatch ./goslurm.sh ann_mnist_digits_serial  #Run serial version for comparison

       #Note for armadillo build
       #Made lib static and issue with MKL on Centos
       #Below changes done in my git, but may need to do if download from
       #http://sourceforge.net/projects/arma/files/armadillo-10.6.2.tar.xz
       #sed -i "s/add_library( armadillo/add_library( armadillo STATIC/" CMakeLists.txt
      #sed -i "s/include(ARMA_FindMKL)/#include(ARMA_FindMKL)/" CMakeLists.txt

