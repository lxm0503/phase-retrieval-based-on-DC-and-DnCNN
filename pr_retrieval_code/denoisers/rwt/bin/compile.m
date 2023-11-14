%    COMPILE compiles the c files and generates mex files.
%

if exist('OCTAVE_VERSION', 'builtin')
  mkoctfile --mex -v -DOCTAVE_MEX_FILE ../mex/mdwt.c   ../lib/src/dwt.c   ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -o omdwt.mex
  mkoctfile --mex -v -DOCTAVE_MEX_FILE ../mex/midwt.c  ../lib/src/idwt.c  ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -o omidwt.mex
  mkoctfile --mex -v -DOCTAVE_MEX_FILE ../mex/mrdwt.c  ../lib/src/rdwt.c  ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -o omrdwt.mex
  mkoctfile --mex -v -DOCTAVE_MEX_FILE ../mex/mirdwt.c ../lib/src/irdwt.c ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -o omirdwt.mex
else
  x = computer();
  if (x(length(x)-1:length(x)) == '64')
    mex -v -largeArrayDims E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/mex/mdwt.c   E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/src/dwt.c   E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/src/init.c E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/src/platform.c -IE:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/inc -outdir E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/bin
    mex -v -largeArrayDims E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/mex/midwt.c  E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/src/idwt.c  E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/src/init.c E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/src/platform.c -IE:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/inc -outdir E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/bin
    mex -v -largeArrayDims E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/mex/mrdwt.c  E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/src/rdwt.c  E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/src/init.c E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/src/platform.c -IE:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/inc -outdir E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/bin
    mex -v -largeArrayDims E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/mex/mirdwt.c E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/src/irdwt.c E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/src/init.c E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/src/platform.c -IE:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/lib/inc -outdir E:/study/prDeep-master/prDeep-master/D-AMP_Toolbox-master/Packages/rwt/bin
  else
    mex -v -compatibleArrayDims ../mex/mdwt.c   ../lib/src/dwt.c   ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -outdir ../bin
    mex -v -compatibleArrayDims ../mex/midwt.c  ../lib/src/idwt.c  ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -outdir ../bin
    mex -v -compatibleArrayDims ../mex/mrdwt.c  ../lib/src/rdwt.c  ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -outdir ../bin
    mex -v -compatibleArrayDims ../mex/mirdwt.c ../lib/src/irdwt.c ../lib/src/init.c ../lib/src/platform.c -I../lib/inc -outdir ../bin
  end
end
