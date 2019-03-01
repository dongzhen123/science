clear;
clc;
% delete Activation.mexw64
% delete conv2d.mexw64
% delete dActivation.mexw64
% delete deconv2d.mexw64
% delete dilconv2d.mexw64
% delete pool.mexw64
% delete uppool.mexw64
mexcuda conv2d.cu
mexcuda deconv2d.cu 
mexcuda dilconv2d.cu 
mexcuda pool.cu 
mexcuda uppool.cu;
mexcuda Activation.cu;
mexcuda dActivation.cu;
mexcuda Adam.cu;
clc;
clc;