git clone git://github.com/yetanotherpassword/cosc3500
cd ~/cosc3500/
unzip mnist.zip
unxz armadillo-10.6.2.tar.xz
tar xvf armadillo-10.6.2.tar
cd armadillo-10.6.2/
#Made lib static and issue with MKL on Centos
#Below changes done in my git, but may need to do if download from 
#http://sourceforge.net/projects/arma/files/armadillo-10.6.2.tar.xz
#sed -i "s/add_library( armadillo/add_library( armadillo STATIC/" CMakeLists.txt
#sed -i "s/include(ARMA_FindMKL)/#include(ARMA_FindMKL)/" CMakeLists.txt
mkdir build
cd build
cmake ..
make
cd ../..
make
sbatch ./goslurm.sh ann_mnist_digits

