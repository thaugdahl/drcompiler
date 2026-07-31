#map = affine_map<(d0, d1) -> (d0 * 64 + d1)>
#map1 = affine_map<(d0) -> (d0 * -2 + 3, 0)>
#map2 = affine_map<(d0) -> (d0 * -2 + 227, 7)>
#map3 = affine_map<(d0, d1) -> (d1 * -2 + 3, 0)>
#map4 = affine_map<(d0, d1) -> (d1 * -2 + 227, 7)>
#map5 = affine_map<(d0, d1) -> (d0 + d1 * 3)>
#map6 = affine_map<(d0, d1) -> (d0 + d1 * 2 - 3)>
#map7 = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>
#map8 = affine_map<(d0, d1) -> (d1, d0 * 2 + d1 - 1)>
#map9 = affine_map<(d0) -> (-d0 + 1, 0)>
#map10 = affine_map<(d0) -> (-d0 + 57, 3)>
#map11 = affine_map<(d0, d1) -> (-d1 + 1, 0)>
#map12 = affine_map<(d0, d1) -> (-d1 + 57, 3)>
#map13 = affine_map<(d0, d1) -> (d0 + d1 * 64)>
#map14 = affine_map<(d0, d1) -> (d0 + d1 - 1)>
#map15 = affine_map<(d0, d1) -> (d0 * 128 + d1)>
#map16 = affine_map<(d0) -> (d0 * -2 + 1, 0)>
#map17 = affine_map<(d0) -> (d0 * -2 + 57, 3)>
#map18 = affine_map<(d0, d1) -> (d1 * -2 + 1, 0)>
#map19 = affine_map<(d0, d1) -> (d1 * -2 + 57, 3)>
#map20 = affine_map<(d0, d1) -> (d0 + d1 * 128)>
#map21 = affine_map<(d0, d1) -> (d0 + d1 * 2 - 1)>
#map22 = affine_map<(d0, d1) -> (d0 * 512 + d1)>
#map23 = affine_map<(d0) -> (d0 * -2, 0)>
#map24 = affine_map<(d0) -> (d0 * -2 + 56, 1)>
#map25 = affine_map<(d0, d1) -> (d1 * -2, 0)>
#map26 = affine_map<(d0, d1) -> (d1 * -2 + 56, 1)>
#map27 = affine_map<(d0, d1) -> (d0 + d1 * 256)>
#map28 = affine_map<(d0, d1) -> (d0 + d1 * 2)>
#map29 = affine_map<(d0) -> (-d0 + 29, 3)>
#map30 = affine_map<(d0, d1) -> (-d1 + 29, 3)>
#map31 = affine_map<(d0, d1) -> (d0 * 256 + d1)>
#map32 = affine_map<(d0) -> (d0 * -2 + 29, 3)>
#map33 = affine_map<(d0, d1) -> (d1 * -2 + 29, 3)>
#map34 = affine_map<(d0, d1) -> (d0 * 1024 + d1)>
#map35 = affine_map<(d0) -> (d0 * -2 + 28, 1)>
#map36 = affine_map<(d0, d1) -> (d1 * -2 + 28, 1)>
#map37 = affine_map<(d0, d1) -> (d0 + d1 * 512)>
#map38 = affine_map<(d0) -> (-d0 + 15, 3)>
#map39 = affine_map<(d0, d1) -> (-d1 + 15, 3)>
#map40 = affine_map<(d0) -> (d0 * -2 + 15, 3)>
#map41 = affine_map<(d0, d1) -> (d1 * -2 + 15, 3)>
#map42 = affine_map<(d0, d1) -> (d0 * 2048 + d1)>
#map43 = affine_map<(d0) -> (d0 * -2 + 14, 1)>
#map44 = affine_map<(d0, d1) -> (d1 * -2 + 14, 1)>
#map45 = affine_map<(d0, d1) -> (d0 + d1 * 1024)>
#map46 = affine_map<(d0) -> (-d0 + 8, 3)>
#map47 = affine_map<(d0, d1) -> (-d1 + 8, 3)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "rn50_s"} {
  memref.global "private" constant @constant_1 : memref<2048x512xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_4 : memref<1x512x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_5 : memref<512x2048xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_7 : memref<2048x512xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_8 : memref<1x512x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_9 : memref<512x2048xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_10 : memref<2048x512xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_12 : memref<1x512x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_13 : memref<512x1024xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_16 : memref<1024x256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_19 : memref<1x256x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_20 : memref<256x1024xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_21 : memref<1024x256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_22 : memref<1x256x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_23 : memref<256x1024xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_24 : memref<1024x256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_25 : memref<1x256x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_26 : memref<256x1024xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_27 : memref<1024x256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_28 : memref<1x256x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_29 : memref<256x1024xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_30 : memref<1024x256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_31 : memref<1x256x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_32 : memref<256x1024xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_33 : memref<1024x256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_35 : memref<1x256x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_36 : memref<256x512xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_39 : memref<512x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_42 : memref<1x128x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_43 : memref<128x512xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_44 : memref<512x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_45 : memref<1x128x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_46 : memref<128x512xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_47 : memref<512x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_48 : memref<1x128x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_49 : memref<128x512xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_50 : memref<512x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_52 : memref<1x128x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_53 : memref<128x256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_56 : memref<256x64xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_59 : memref<1x64x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_60 : memref<64x256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_61 : memref<256x64xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_62 : memref<1x64x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_63 : memref<64x256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_64 : memref<256x64xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_65 : memref<256x64xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_66 : memref<1x64x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_67 : memref<64x64xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_68 : memref<2048x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_69 : memref<2048x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_70 : memref<512xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_71 : memref<512x512x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_72 : memref<2048x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_73 : memref<2048x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_74 : memref<512xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_75 : memref<512x512x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_76 : memref<2048x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_77 : memref<2048x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_78 : memref<512xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_79 : memref<512x512x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_80 : memref<1024x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_81 : memref<1024x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_82 : memref<256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_83 : memref<256x256x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_84 : memref<1024x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_85 : memref<1024x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_86 : memref<256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_87 : memref<256x256x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_88 : memref<1024x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_89 : memref<1024x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_90 : memref<256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_91 : memref<256x256x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_92 : memref<1024x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_93 : memref<1024x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_94 : memref<256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_95 : memref<256x256x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_96 : memref<1024x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_97 : memref<1024x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_98 : memref<256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_99 : memref<256x256x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_100 : memref<1024x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_101 : memref<1024x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_102 : memref<256xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_103 : memref<256x256x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_104 : memref<512x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_105 : memref<512x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_106 : memref<128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_107 : memref<128x128x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_108 : memref<512x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_109 : memref<512x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_110 : memref<128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_111 : memref<128x128x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_112 : memref<512x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_113 : memref<512x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_114 : memref<128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_115 : memref<128x128x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_116 : memref<512x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_117 : memref<512x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_118 : memref<128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_119 : memref<128x128x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_120 : memref<256x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_121 : memref<256x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_122 : memref<64xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_123 : memref<64x64x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_124 : memref<256x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_125 : memref<256x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_126 : memref<64xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_127 : memref<64x64x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_128 : memref<256x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_129 : memref<256x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_130 : memref<64xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_131 : memref<64x64x3x3xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_132 : memref<64x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_133 : memref<64x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_134 : memref<64xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_135 : memref<64x3x7x7xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_136 : memref<3x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_137 : memref<512x256x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_138 : memref<1024x512x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_139 : memref<2048x1024x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_141 : memref<1000x2048xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_142 : memref<1000xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_143 : memref<3x1x1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  func.func @main_graph(%arg0: memref<1x3x224x224xf32> {onnx.dim_params = "0:N", onnx.name = "data"}) -> (memref<1x1000xf32> {onnx.dim_params = "0:N", onnx.name = "resnetv24_dense0_fwd"}) attributes {llvm.emit_c_interface} {
    %cst = arith.constant 4.900000e+01 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c112 = arith.constant 112 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %0 = memref.get_global @constant_1 : memref<2048x512xf32>
    %1 = memref.get_global @constant_4 : memref<1x512x1xf32>
    %2 = memref.get_global @constant_5 : memref<512x2048xf32>
    %3 = memref.get_global @constant_7 : memref<2048x512xf32>
    %4 = memref.get_global @constant_8 : memref<1x512x1xf32>
    %5 = memref.get_global @constant_9 : memref<512x2048xf32>
    %6 = memref.get_global @constant_10 : memref<2048x512xf32>
    %7 = memref.get_global @constant_12 : memref<1x512x1xf32>
    %8 = memref.get_global @constant_13 : memref<512x1024xf32>
    %9 = memref.get_global @constant_16 : memref<1024x256xf32>
    %10 = memref.get_global @constant_19 : memref<1x256x1xf32>
    %11 = memref.get_global @constant_20 : memref<256x1024xf32>
    %12 = memref.get_global @constant_21 : memref<1024x256xf32>
    %13 = memref.get_global @constant_22 : memref<1x256x1xf32>
    %14 = memref.get_global @constant_23 : memref<256x1024xf32>
    %15 = memref.get_global @constant_24 : memref<1024x256xf32>
    %16 = memref.get_global @constant_25 : memref<1x256x1xf32>
    %17 = memref.get_global @constant_26 : memref<256x1024xf32>
    %18 = memref.get_global @constant_27 : memref<1024x256xf32>
    %19 = memref.get_global @constant_28 : memref<1x256x1xf32>
    %20 = memref.get_global @constant_29 : memref<256x1024xf32>
    %21 = memref.get_global @constant_30 : memref<1024x256xf32>
    %22 = memref.get_global @constant_31 : memref<1x256x1xf32>
    %23 = memref.get_global @constant_32 : memref<256x1024xf32>
    %24 = memref.get_global @constant_33 : memref<1024x256xf32>
    %25 = memref.get_global @constant_35 : memref<1x256x1xf32>
    %26 = memref.get_global @constant_36 : memref<256x512xf32>
    %27 = memref.get_global @constant_39 : memref<512x128xf32>
    %28 = memref.get_global @constant_42 : memref<1x128x1xf32>
    %29 = memref.get_global @constant_43 : memref<128x512xf32>
    %30 = memref.get_global @constant_44 : memref<512x128xf32>
    %31 = memref.get_global @constant_45 : memref<1x128x1xf32>
    %32 = memref.get_global @constant_46 : memref<128x512xf32>
    %33 = memref.get_global @constant_47 : memref<512x128xf32>
    %34 = memref.get_global @constant_48 : memref<1x128x1xf32>
    %35 = memref.get_global @constant_49 : memref<128x512xf32>
    %36 = memref.get_global @constant_50 : memref<512x128xf32>
    %37 = memref.get_global @constant_52 : memref<1x128x1xf32>
    %38 = memref.get_global @constant_53 : memref<128x256xf32>
    %39 = memref.get_global @constant_56 : memref<256x64xf32>
    %40 = memref.get_global @constant_59 : memref<1x64x1xf32>
    %41 = memref.get_global @constant_60 : memref<64x256xf32>
    %42 = memref.get_global @constant_61 : memref<256x64xf32>
    %43 = memref.get_global @constant_62 : memref<1x64x1xf32>
    %44 = memref.get_global @constant_63 : memref<64x256xf32>
    %45 = memref.get_global @constant_64 : memref<256x64xf32>
    %46 = memref.get_global @constant_65 : memref<256x64xf32>
    %47 = memref.get_global @constant_66 : memref<1x64x1xf32>
    %48 = memref.get_global @constant_67 : memref<64x64xf32>
    %49 = memref.get_global @constant_68 : memref<2048x1x1xf32>
    %50 = memref.get_global @constant_69 : memref<2048x1x1xf32>
    %51 = memref.get_global @constant_70 : memref<512xf32>
    %52 = memref.get_global @constant_71 : memref<512x512x3x3xf32>
    %53 = memref.get_global @constant_72 : memref<2048x1x1xf32>
    %54 = memref.get_global @constant_73 : memref<2048x1x1xf32>
    %55 = memref.get_global @constant_74 : memref<512xf32>
    %56 = memref.get_global @constant_75 : memref<512x512x3x3xf32>
    %57 = memref.get_global @constant_76 : memref<2048x1x1xf32>
    %58 = memref.get_global @constant_77 : memref<2048x1x1xf32>
    %59 = memref.get_global @constant_78 : memref<512xf32>
    %60 = memref.get_global @constant_79 : memref<512x512x3x3xf32>
    %61 = memref.get_global @constant_80 : memref<1024x1x1xf32>
    %62 = memref.get_global @constant_81 : memref<1024x1x1xf32>
    %63 = memref.get_global @constant_82 : memref<256xf32>
    %64 = memref.get_global @constant_83 : memref<256x256x3x3xf32>
    %65 = memref.get_global @constant_84 : memref<1024x1x1xf32>
    %66 = memref.get_global @constant_85 : memref<1024x1x1xf32>
    %67 = memref.get_global @constant_86 : memref<256xf32>
    %68 = memref.get_global @constant_87 : memref<256x256x3x3xf32>
    %69 = memref.get_global @constant_88 : memref<1024x1x1xf32>
    %70 = memref.get_global @constant_89 : memref<1024x1x1xf32>
    %71 = memref.get_global @constant_90 : memref<256xf32>
    %72 = memref.get_global @constant_91 : memref<256x256x3x3xf32>
    %73 = memref.get_global @constant_92 : memref<1024x1x1xf32>
    %74 = memref.get_global @constant_93 : memref<1024x1x1xf32>
    %75 = memref.get_global @constant_94 : memref<256xf32>
    %76 = memref.get_global @constant_95 : memref<256x256x3x3xf32>
    %77 = memref.get_global @constant_96 : memref<1024x1x1xf32>
    %78 = memref.get_global @constant_97 : memref<1024x1x1xf32>
    %79 = memref.get_global @constant_98 : memref<256xf32>
    %80 = memref.get_global @constant_99 : memref<256x256x3x3xf32>
    %81 = memref.get_global @constant_100 : memref<1024x1x1xf32>
    %82 = memref.get_global @constant_101 : memref<1024x1x1xf32>
    %83 = memref.get_global @constant_102 : memref<256xf32>
    %84 = memref.get_global @constant_103 : memref<256x256x3x3xf32>
    %85 = memref.get_global @constant_104 : memref<512x1x1xf32>
    %86 = memref.get_global @constant_105 : memref<512x1x1xf32>
    %87 = memref.get_global @constant_106 : memref<128xf32>
    %88 = memref.get_global @constant_107 : memref<128x128x3x3xf32>
    %89 = memref.get_global @constant_108 : memref<512x1x1xf32>
    %90 = memref.get_global @constant_109 : memref<512x1x1xf32>
    %91 = memref.get_global @constant_110 : memref<128xf32>
    %92 = memref.get_global @constant_111 : memref<128x128x3x3xf32>
    %93 = memref.get_global @constant_112 : memref<512x1x1xf32>
    %94 = memref.get_global @constant_113 : memref<512x1x1xf32>
    %95 = memref.get_global @constant_114 : memref<128xf32>
    %96 = memref.get_global @constant_115 : memref<128x128x3x3xf32>
    %97 = memref.get_global @constant_116 : memref<512x1x1xf32>
    %98 = memref.get_global @constant_117 : memref<512x1x1xf32>
    %99 = memref.get_global @constant_118 : memref<128xf32>
    %100 = memref.get_global @constant_119 : memref<128x128x3x3xf32>
    %101 = memref.get_global @constant_120 : memref<256x1x1xf32>
    %102 = memref.get_global @constant_121 : memref<256x1x1xf32>
    %103 = memref.get_global @constant_122 : memref<64xf32>
    %104 = memref.get_global @constant_123 : memref<64x64x3x3xf32>
    %105 = memref.get_global @constant_124 : memref<256x1x1xf32>
    %106 = memref.get_global @constant_125 : memref<256x1x1xf32>
    %107 = memref.get_global @constant_126 : memref<64xf32>
    %108 = memref.get_global @constant_127 : memref<64x64x3x3xf32>
    %109 = memref.get_global @constant_128 : memref<256x1x1xf32>
    %110 = memref.get_global @constant_129 : memref<256x1x1xf32>
    %111 = memref.get_global @constant_130 : memref<64xf32>
    %112 = memref.get_global @constant_131 : memref<64x64x3x3xf32>
    %113 = memref.get_global @constant_132 : memref<64x1x1xf32>
    %114 = memref.get_global @constant_133 : memref<64x1x1xf32>
    %115 = memref.get_global @constant_134 : memref<64xf32>
    %116 = memref.get_global @constant_135 : memref<64x3x7x7xf32>
    %117 = memref.get_global @constant_136 : memref<3x1x1xf32>
    %118 = memref.get_global @constant_137 : memref<512x256x1x1xf32>
    %119 = memref.get_global @constant_138 : memref<1024x512x1x1xf32>
    %120 = memref.get_global @constant_139 : memref<2048x1024x1x1xf32>
    %121 = memref.get_global @constant_141 : memref<1000x2048xf32>
    %122 = memref.get_global @constant_142 : memref<1000xf32>
    %123 = memref.get_global @constant_143 : memref<3x1x1xf32>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x3x224x224xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 3 {
        affine.for %arg3 = 0 to 224 {
          affine.for %arg4 = 0 to 224 {
            %124 = affine.load %arg0[%c0, %arg2, %arg3, %arg4] : memref<1x3x224x224xf32>
            %125 = affine.load %123[%arg2, %c0, %c0] : memref<3x1x1xf32>
            %126 = arith.mulf %124, %125 : f32
            %127 = affine.load %117[%arg2, %c0, %c0] : memref<3x1x1xf32>
            %128 = arith.addf %126, %127 : f32
            affine.store %128, %alloc[%arg1, %arg2, %arg3, %arg4] : memref<1x3x224x224xf32>
          }
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 16 : i64} : memref<1x64x112x112xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 64 {
          %124 = affine.apply #map(%arg2, %arg3)
          affine.for %arg4 = 0 to 112 {
            affine.for %arg5 = 0 to 112 {
              %125 = affine.for %arg6 = 0 to 3 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map1(%arg4) to min #map2(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map3(%arg4, %arg5) to min #map4(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map5(%arg6, %arg2)
                    %131 = affine.apply #map6(%arg8, %arg4)
                    %132 = affine.apply #map6(%arg10, %arg5)
                    %133 = affine.load %alloc[%arg1, %130, %131, %132] : memref<1x3x224x224xf32>
                    %134 = affine.load %116[%124, %arg6, %arg8, %arg10] : memref<64x3x7x7xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %115[%124] : memref<64xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_2[%arg1, %124, %arg4, %arg5] : memref<1x64x112x112xf32>
            }
          }
        }
      }
    }
    %alloc_3 = memref.alloc() {alignment = 16 : i64} : memref<1x64x112x112xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 112 {
          affine.for %arg4 = 0 to 112 {
            %124 = affine.load %alloc_2[%arg1, %arg2, %arg3, %arg4] : memref<1x64x112x112xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_3[%arg1, %arg2, %arg3, %arg4] : memref<1x64x112x112xf32>
          }
        }
      }
    }
    %alloc_4 = memref.alloc() {alignment = 16 : i64} : memref<1x64x56x56xf32>
    %alloca = memref.alloca() : memref<f32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 56 {
          affine.for %arg4 = 0 to 56 {
            affine.store %cst_0, %alloca[] : memref<f32>
            affine.for %arg5 = 0 to min #map7(%arg3)[%c112, %c3, %c1, %c2, %c1] {
              affine.for %arg6 = 0 to min #map7(%arg4)[%c112, %c3, %c1, %c2, %c1] {
                %125 = affine.max #map8(%arg3, %arg5)
                %126 = affine.max #map8(%arg4, %arg6)
                %127 = memref.load %alloc_3[%arg1, %arg2, %125, %126] : memref<1x64x112x112xf32>
                %128 = affine.load %alloca[] : memref<f32>
                %129 = arith.maxf %128, %127 : f32
                affine.store %129, %alloca[] : memref<f32>
              }
            }
            %124 = affine.load %alloca[] : memref<f32>
            affine.store %124, %alloc_4[%arg1, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
          }
        }
      }
    }
    %alloc_5 = memref.alloc() {alignment = 16 : i64} : memref<1x64x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 56 {
          affine.for %arg4 = 0 to 56 {
            %124 = affine.load %alloc_4[%c0, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
            %125 = affine.load %114[%arg2, %c0, %c0] : memref<64x1x1xf32>
            %126 = arith.mulf %124, %125 : f32
            %127 = affine.load %113[%arg2, %c0, %c0] : memref<64x1x1xf32>
            %128 = arith.addf %126, %127 : f32
            %129 = arith.maxf %128, %cst_1 : f32
            affine.store %129, %alloc_5[%arg1, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
          }
        }
      }
    }
    %reinterpret_cast = memref.reinterpret_cast %alloc_5 to offset: [0], sizes: [1, 64, 3136], strides: [200704, 3136, 1] : memref<1x64x56x56xf32> to memref<1x64x3136xf32>
    %alloc_6 = memref.alloc() {alignment = 16 : i64} : memref<1x64x3136xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 3136 {
          %124 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %48[%arg2, %arg4] : memref<64x64xf32>
            %126 = affine.load %reinterpret_cast[%arg1, %arg4, %arg3] : memref<1x64x3136xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_6[%arg1, %arg2, %arg3] : memref<1x64x3136xf32>
        }
      }
    }
    %alloc_7 = memref.alloc() {alignment = 16 : i64} : memref<1x64x3136xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 3136 {
          %124 = affine.load %alloc_6[%c0, %arg2, %arg3] : memref<1x64x3136xf32>
          %125 = affine.load %47[%c0, %arg2, %c0] : memref<1x64x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_7[%arg1, %arg2, %arg3] : memref<1x64x3136xf32>
        }
      }
    }
    %reinterpret_cast_8 = memref.reinterpret_cast %alloc_7 to offset: [0], sizes: [1, 64, 56, 56], strides: [200704, 3136, 56, 1] : memref<1x64x3136xf32> to memref<1x64x56x56xf32>
    %alloc_9 = memref.alloc() {alignment = 16 : i64} : memref<1x64x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 56 {
          affine.for %arg4 = 0 to 56 {
            %124 = affine.load %reinterpret_cast_8[%arg1, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_9[%arg1, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
          }
        }
      }
    }
    %alloc_10 = memref.alloc() {alignment = 16 : i64} : memref<1x64x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 64 {
          %124 = affine.apply #map(%arg2, %arg3)
          affine.for %arg4 = 0 to 56 {
            affine.for %arg5 = 0 to 56 {
              %125 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map9(%arg4) to min #map10(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map11(%arg4, %arg5) to min #map12(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map13(%arg6, %arg2)
                    %131 = affine.apply #map14(%arg8, %arg4)
                    %132 = affine.apply #map14(%arg10, %arg5)
                    %133 = affine.load %alloc_9[%arg1, %130, %131, %132] : memref<1x64x56x56xf32>
                    %134 = affine.load %112[%124, %arg6, %arg8, %arg10] : memref<64x64x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %111[%124] : memref<64xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_10[%arg1, %124, %arg4, %arg5] : memref<1x64x56x56xf32>
            }
          }
        }
      }
    }
    %alloc_11 = memref.alloc() {alignment = 16 : i64} : memref<1x64x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 56 {
          affine.for %arg4 = 0 to 56 {
            %124 = affine.load %alloc_10[%arg1, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_11[%arg1, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
          }
        }
      }
    }
    %reinterpret_cast_12 = memref.reinterpret_cast %alloc_11 to offset: [0], sizes: [1, 64, 3136], strides: [200704, 3136, 1] : memref<1x64x56x56xf32> to memref<1x64x3136xf32>
    %alloc_13 = memref.alloc() {alignment = 16 : i64} : memref<1x256x3136xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 3136 {
          %124 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %46[%arg2, %arg4] : memref<256x64xf32>
            %126 = affine.load %reinterpret_cast_12[%arg1, %arg4, %arg3] : memref<1x64x3136xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_13[%arg1, %arg2, %arg3] : memref<1x256x3136xf32>
        }
      }
    }
    %reinterpret_cast_14 = memref.reinterpret_cast %alloc_13 to offset: [0], sizes: [1, 256, 56, 56], strides: [802816, 3136, 56, 1] : memref<1x256x3136xf32> to memref<1x256x56x56xf32>
    %reinterpret_cast_15 = memref.reinterpret_cast %alloc_5 to offset: [0], sizes: [1, 64, 3136], strides: [200704, 3136, 1] : memref<1x64x56x56xf32> to memref<1x64x3136xf32>
    %alloc_16 = memref.alloc() {alignment = 16 : i64} : memref<1x256x3136xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 3136 {
          %124 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %45[%arg2, %arg4] : memref<256x64xf32>
            %126 = affine.load %reinterpret_cast_15[%arg1, %arg4, %arg3] : memref<1x64x3136xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_16[%arg1, %arg2, %arg3] : memref<1x256x3136xf32>
        }
      }
    }
    %reinterpret_cast_17 = memref.reinterpret_cast %alloc_16 to offset: [0], sizes: [1, 256, 56, 56], strides: [802816, 3136, 56, 1] : memref<1x256x3136xf32> to memref<1x256x56x56xf32>
    %alloc_18 = memref.alloc() {alignment = 16 : i64} : memref<1x256x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 56 {
          affine.for %arg4 = 0 to 56 {
            %124 = affine.load %reinterpret_cast_14[%c0, %arg2, %arg3, %arg4] : memref<1x256x56x56xf32>
            %125 = affine.load %reinterpret_cast_17[%c0, %arg2, %arg3, %arg4] : memref<1x256x56x56xf32>
            %126 = arith.addf %124, %125 : f32
            affine.store %126, %alloc_18[%arg1, %arg2, %arg3, %arg4] : memref<1x256x56x56xf32>
          }
        }
      }
    }
    %alloc_19 = memref.alloc() {alignment = 16 : i64} : memref<1x256x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 56 {
          affine.for %arg4 = 0 to 56 {
            %124 = affine.load %alloc_18[%c0, %arg2, %arg3, %arg4] : memref<1x256x56x56xf32>
            %125 = affine.load %110[%arg2, %c0, %c0] : memref<256x1x1xf32>
            %126 = arith.mulf %124, %125 : f32
            %127 = affine.load %109[%arg2, %c0, %c0] : memref<256x1x1xf32>
            %128 = arith.addf %126, %127 : f32
            %129 = arith.maxf %128, %cst_1 : f32
            affine.store %129, %alloc_19[%arg1, %arg2, %arg3, %arg4] : memref<1x256x56x56xf32>
          }
        }
      }
    }
    %reinterpret_cast_20 = memref.reinterpret_cast %alloc_19 to offset: [0], sizes: [1, 256, 3136], strides: [802816, 3136, 1] : memref<1x256x56x56xf32> to memref<1x256x3136xf32>
    %alloc_21 = memref.alloc() {alignment = 16 : i64} : memref<1x64x3136xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 3136 {
          %124 = affine.for %arg4 = 0 to 256 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %44[%arg2, %arg4] : memref<64x256xf32>
            %126 = affine.load %reinterpret_cast_20[%arg1, %arg4, %arg3] : memref<1x256x3136xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_21[%arg1, %arg2, %arg3] : memref<1x64x3136xf32>
        }
      }
    }
    %alloc_22 = memref.alloc() {alignment = 16 : i64} : memref<1x64x3136xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 3136 {
          %124 = affine.load %alloc_21[%c0, %arg2, %arg3] : memref<1x64x3136xf32>
          %125 = affine.load %43[%c0, %arg2, %c0] : memref<1x64x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_22[%arg1, %arg2, %arg3] : memref<1x64x3136xf32>
        }
      }
    }
    %reinterpret_cast_23 = memref.reinterpret_cast %alloc_22 to offset: [0], sizes: [1, 64, 56, 56], strides: [200704, 3136, 56, 1] : memref<1x64x3136xf32> to memref<1x64x56x56xf32>
    %alloc_24 = memref.alloc() {alignment = 16 : i64} : memref<1x64x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 56 {
          affine.for %arg4 = 0 to 56 {
            %124 = affine.load %reinterpret_cast_23[%arg1, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_24[%arg1, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
          }
        }
      }
    }
    %alloc_25 = memref.alloc() {alignment = 16 : i64} : memref<1x64x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 64 {
          %124 = affine.apply #map(%arg2, %arg3)
          affine.for %arg4 = 0 to 56 {
            affine.for %arg5 = 0 to 56 {
              %125 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map9(%arg4) to min #map10(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map11(%arg4, %arg5) to min #map12(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map13(%arg6, %arg2)
                    %131 = affine.apply #map14(%arg8, %arg4)
                    %132 = affine.apply #map14(%arg10, %arg5)
                    %133 = affine.load %alloc_24[%arg1, %130, %131, %132] : memref<1x64x56x56xf32>
                    %134 = affine.load %108[%124, %arg6, %arg8, %arg10] : memref<64x64x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %107[%124] : memref<64xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_25[%arg1, %124, %arg4, %arg5] : memref<1x64x56x56xf32>
            }
          }
        }
      }
    }
    %alloc_26 = memref.alloc() {alignment = 16 : i64} : memref<1x64x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 56 {
          affine.for %arg4 = 0 to 56 {
            %124 = affine.load %alloc_25[%arg1, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_26[%arg1, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
          }
        }
      }
    }
    %reinterpret_cast_27 = memref.reinterpret_cast %alloc_26 to offset: [0], sizes: [1, 64, 3136], strides: [200704, 3136, 1] : memref<1x64x56x56xf32> to memref<1x64x3136xf32>
    %alloc_28 = memref.alloc() {alignment = 16 : i64} : memref<1x256x3136xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 3136 {
          %124 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %42[%arg2, %arg4] : memref<256x64xf32>
            %126 = affine.load %reinterpret_cast_27[%arg1, %arg4, %arg3] : memref<1x64x3136xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_28[%arg1, %arg2, %arg3] : memref<1x256x3136xf32>
        }
      }
    }
    %reinterpret_cast_29 = memref.reinterpret_cast %alloc_28 to offset: [0], sizes: [1, 256, 56, 56], strides: [802816, 3136, 56, 1] : memref<1x256x3136xf32> to memref<1x256x56x56xf32>
    %alloc_30 = memref.alloc() {alignment = 16 : i64} : memref<1x256x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 56 {
          affine.for %arg4 = 0 to 56 {
            %124 = affine.load %reinterpret_cast_29[%c0, %arg2, %arg3, %arg4] : memref<1x256x56x56xf32>
            %125 = affine.load %alloc_18[%c0, %arg2, %arg3, %arg4] : memref<1x256x56x56xf32>
            %126 = arith.addf %124, %125 : f32
            affine.store %126, %alloc_30[%arg1, %arg2, %arg3, %arg4] : memref<1x256x56x56xf32>
          }
        }
      }
    }
    %alloc_31 = memref.alloc() {alignment = 16 : i64} : memref<1x256x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 56 {
          affine.for %arg4 = 0 to 56 {
            %124 = affine.load %alloc_30[%c0, %arg2, %arg3, %arg4] : memref<1x256x56x56xf32>
            %125 = affine.load %106[%arg2, %c0, %c0] : memref<256x1x1xf32>
            %126 = arith.mulf %124, %125 : f32
            %127 = affine.load %105[%arg2, %c0, %c0] : memref<256x1x1xf32>
            %128 = arith.addf %126, %127 : f32
            %129 = arith.maxf %128, %cst_1 : f32
            affine.store %129, %alloc_31[%arg1, %arg2, %arg3, %arg4] : memref<1x256x56x56xf32>
          }
        }
      }
    }
    %reinterpret_cast_32 = memref.reinterpret_cast %alloc_31 to offset: [0], sizes: [1, 256, 3136], strides: [802816, 3136, 1] : memref<1x256x56x56xf32> to memref<1x256x3136xf32>
    %alloc_33 = memref.alloc() {alignment = 16 : i64} : memref<1x64x3136xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 3136 {
          %124 = affine.for %arg4 = 0 to 256 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %41[%arg2, %arg4] : memref<64x256xf32>
            %126 = affine.load %reinterpret_cast_32[%arg1, %arg4, %arg3] : memref<1x256x3136xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_33[%arg1, %arg2, %arg3] : memref<1x64x3136xf32>
        }
      }
    }
    %alloc_34 = memref.alloc() {alignment = 16 : i64} : memref<1x64x3136xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 3136 {
          %124 = affine.load %alloc_33[%c0, %arg2, %arg3] : memref<1x64x3136xf32>
          %125 = affine.load %40[%c0, %arg2, %c0] : memref<1x64x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_34[%arg1, %arg2, %arg3] : memref<1x64x3136xf32>
        }
      }
    }
    %reinterpret_cast_35 = memref.reinterpret_cast %alloc_34 to offset: [0], sizes: [1, 64, 56, 56], strides: [200704, 3136, 56, 1] : memref<1x64x3136xf32> to memref<1x64x56x56xf32>
    %alloc_36 = memref.alloc() {alignment = 16 : i64} : memref<1x64x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 56 {
          affine.for %arg4 = 0 to 56 {
            %124 = affine.load %reinterpret_cast_35[%arg1, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_36[%arg1, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
          }
        }
      }
    }
    %alloc_37 = memref.alloc() {alignment = 16 : i64} : memref<1x64x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 64 {
          %124 = affine.apply #map(%arg2, %arg3)
          affine.for %arg4 = 0 to 56 {
            affine.for %arg5 = 0 to 56 {
              %125 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map9(%arg4) to min #map10(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map11(%arg4, %arg5) to min #map12(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map13(%arg6, %arg2)
                    %131 = affine.apply #map14(%arg8, %arg4)
                    %132 = affine.apply #map14(%arg10, %arg5)
                    %133 = affine.load %alloc_36[%arg1, %130, %131, %132] : memref<1x64x56x56xf32>
                    %134 = affine.load %104[%124, %arg6, %arg8, %arg10] : memref<64x64x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %103[%124] : memref<64xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_37[%arg1, %124, %arg4, %arg5] : memref<1x64x56x56xf32>
            }
          }
        }
      }
    }
    %alloc_38 = memref.alloc() {alignment = 16 : i64} : memref<1x64x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 64 {
        affine.for %arg3 = 0 to 56 {
          affine.for %arg4 = 0 to 56 {
            %124 = affine.load %alloc_37[%arg1, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_38[%arg1, %arg2, %arg3, %arg4] : memref<1x64x56x56xf32>
          }
        }
      }
    }
    %reinterpret_cast_39 = memref.reinterpret_cast %alloc_38 to offset: [0], sizes: [1, 64, 3136], strides: [200704, 3136, 1] : memref<1x64x56x56xf32> to memref<1x64x3136xf32>
    %alloc_40 = memref.alloc() {alignment = 16 : i64} : memref<1x256x3136xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 3136 {
          %124 = affine.for %arg4 = 0 to 64 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %39[%arg2, %arg4] : memref<256x64xf32>
            %126 = affine.load %reinterpret_cast_39[%arg1, %arg4, %arg3] : memref<1x64x3136xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_40[%arg1, %arg2, %arg3] : memref<1x256x3136xf32>
        }
      }
    }
    %reinterpret_cast_41 = memref.reinterpret_cast %alloc_40 to offset: [0], sizes: [1, 256, 56, 56], strides: [802816, 3136, 56, 1] : memref<1x256x3136xf32> to memref<1x256x56x56xf32>
    %alloc_42 = memref.alloc() {alignment = 16 : i64} : memref<1x256x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 56 {
          affine.for %arg4 = 0 to 56 {
            %124 = affine.load %reinterpret_cast_41[%c0, %arg2, %arg3, %arg4] : memref<1x256x56x56xf32>
            %125 = affine.load %alloc_30[%c0, %arg2, %arg3, %arg4] : memref<1x256x56x56xf32>
            %126 = arith.addf %124, %125 : f32
            %127 = affine.load %102[%arg2, %c0, %c0] : memref<256x1x1xf32>
            %128 = arith.mulf %126, %127 : f32
            %129 = affine.load %101[%arg2, %c0, %c0] : memref<256x1x1xf32>
            %130 = arith.addf %128, %129 : f32
            %131 = arith.maxf %130, %cst_1 : f32
            affine.store %131, %alloc_42[%arg1, %arg2, %arg3, %arg4] : memref<1x256x56x56xf32>
          }
        }
      }
    }
    %reinterpret_cast_43 = memref.reinterpret_cast %alloc_42 to offset: [0], sizes: [1, 256, 3136], strides: [802816, 3136, 1] : memref<1x256x56x56xf32> to memref<1x256x3136xf32>
    %alloc_44 = memref.alloc() {alignment = 16 : i64} : memref<1x128x3136xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 3136 {
          %124 = affine.for %arg4 = 0 to 256 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %38[%arg2, %arg4] : memref<128x256xf32>
            %126 = affine.load %reinterpret_cast_43[%arg1, %arg4, %arg3] : memref<1x256x3136xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_44[%arg1, %arg2, %arg3] : memref<1x128x3136xf32>
        }
      }
    }
    %alloc_45 = memref.alloc() {alignment = 16 : i64} : memref<1x128x3136xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 3136 {
          %124 = affine.load %alloc_44[%c0, %arg2, %arg3] : memref<1x128x3136xf32>
          %125 = affine.load %37[%c0, %arg2, %c0] : memref<1x128x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_45[%arg1, %arg2, %arg3] : memref<1x128x3136xf32>
        }
      }
    }
    %reinterpret_cast_46 = memref.reinterpret_cast %alloc_45 to offset: [0], sizes: [1, 128, 56, 56], strides: [401408, 3136, 56, 1] : memref<1x128x3136xf32> to memref<1x128x56x56xf32>
    %alloc_47 = memref.alloc() {alignment = 16 : i64} : memref<1x128x56x56xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 56 {
          affine.for %arg4 = 0 to 56 {
            %124 = affine.load %reinterpret_cast_46[%arg1, %arg2, %arg3, %arg4] : memref<1x128x56x56xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_47[%arg1, %arg2, %arg3, %arg4] : memref<1x128x56x56xf32>
          }
        }
      }
    }
    %alloc_48 = memref.alloc() {alignment = 16 : i64} : memref<1x128x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 128 {
          %124 = affine.apply #map15(%arg2, %arg3)
          affine.for %arg4 = 0 to 28 {
            affine.for %arg5 = 0 to 28 {
              %125 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map16(%arg4) to min #map17(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map18(%arg4, %arg5) to min #map19(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map20(%arg6, %arg2)
                    %131 = affine.apply #map21(%arg8, %arg4)
                    %132 = affine.apply #map21(%arg10, %arg5)
                    %133 = affine.load %alloc_47[%arg1, %130, %131, %132] : memref<1x128x56x56xf32>
                    %134 = affine.load %100[%124, %arg6, %arg8, %arg10] : memref<128x128x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %99[%124] : memref<128xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_48[%arg1, %124, %arg4, %arg5] : memref<1x128x28x28xf32>
            }
          }
        }
      }
    }
    %alloc_49 = memref.alloc() {alignment = 16 : i64} : memref<1x128x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %alloc_48[%arg1, %arg2, %arg3, %arg4] : memref<1x128x28x28xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_49[%arg1, %arg2, %arg3, %arg4] : memref<1x128x28x28xf32>
          }
        }
      }
    }
    %reinterpret_cast_50 = memref.reinterpret_cast %alloc_49 to offset: [0], sizes: [1, 128, 784], strides: [100352, 784, 1] : memref<1x128x28x28xf32> to memref<1x128x784xf32>
    %alloc_51 = memref.alloc() {alignment = 16 : i64} : memref<1x512x784xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 784 {
          %124 = affine.for %arg4 = 0 to 128 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %36[%arg2, %arg4] : memref<512x128xf32>
            %126 = affine.load %reinterpret_cast_50[%arg1, %arg4, %arg3] : memref<1x128x784xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_51[%arg1, %arg2, %arg3] : memref<1x512x784xf32>
        }
      }
    }
    %reinterpret_cast_52 = memref.reinterpret_cast %alloc_51 to offset: [0], sizes: [1, 512, 28, 28], strides: [401408, 784, 28, 1] : memref<1x512x784xf32> to memref<1x512x28x28xf32>
    %alloc_53 = memref.alloc() {alignment = 16 : i64} : memref<1x512x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 512 {
          %124 = affine.apply #map22(%arg2, %arg3)
          affine.for %arg4 = 0 to 28 {
            affine.for %arg5 = 0 to 28 {
              %125 = affine.for %arg6 = 0 to 256 iter_args(%arg7 = %cst_1) -> (f32) {
                %126 = affine.for %arg8 = max #map23(%arg4) to min #map24(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %127 = affine.for %arg10 = max #map25(%arg4, %arg5) to min #map26(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %128 = affine.apply #map27(%arg6, %arg2)
                    %129 = affine.apply #map28(%arg8, %arg4)
                    %130 = affine.apply #map28(%arg10, %arg5)
                    %131 = affine.load %alloc_42[%arg1, %128, %129, %130] : memref<1x256x56x56xf32>
                    %132 = affine.load %118[%124, %arg6, %arg8, %arg10] : memref<512x256x1x1xf32>
                    %133 = arith.mulf %131, %132 : f32
                    %134 = arith.addf %arg11, %133 : f32
                    affine.yield %134 : f32
                  }
                  affine.yield %127 : f32
                }
                affine.yield %126 : f32
              }
              affine.store %125, %alloc_53[%arg1, %124, %arg4, %arg5] : memref<1x512x28x28xf32>
            }
          }
        }
      }
    }
    %alloc_54 = memref.alloc() {alignment = 16 : i64} : memref<1x512x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %reinterpret_cast_52[%c0, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
            %125 = affine.load %alloc_53[%c0, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
            %126 = arith.addf %124, %125 : f32
            affine.store %126, %alloc_54[%arg1, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
          }
        }
      }
    }
    %alloc_55 = memref.alloc() {alignment = 16 : i64} : memref<1x512x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %alloc_54[%c0, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
            %125 = affine.load %98[%arg2, %c0, %c0] : memref<512x1x1xf32>
            %126 = arith.mulf %124, %125 : f32
            %127 = affine.load %97[%arg2, %c0, %c0] : memref<512x1x1xf32>
            %128 = arith.addf %126, %127 : f32
            %129 = arith.maxf %128, %cst_1 : f32
            affine.store %129, %alloc_55[%arg1, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
          }
        }
      }
    }
    %reinterpret_cast_56 = memref.reinterpret_cast %alloc_55 to offset: [0], sizes: [1, 512, 784], strides: [401408, 784, 1] : memref<1x512x28x28xf32> to memref<1x512x784xf32>
    %alloc_57 = memref.alloc() {alignment = 16 : i64} : memref<1x128x784xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 784 {
          %124 = affine.for %arg4 = 0 to 512 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %35[%arg2, %arg4] : memref<128x512xf32>
            %126 = affine.load %reinterpret_cast_56[%arg1, %arg4, %arg3] : memref<1x512x784xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_57[%arg1, %arg2, %arg3] : memref<1x128x784xf32>
        }
      }
    }
    %alloc_58 = memref.alloc() {alignment = 16 : i64} : memref<1x128x784xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 784 {
          %124 = affine.load %alloc_57[%c0, %arg2, %arg3] : memref<1x128x784xf32>
          %125 = affine.load %34[%c0, %arg2, %c0] : memref<1x128x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_58[%arg1, %arg2, %arg3] : memref<1x128x784xf32>
        }
      }
    }
    %reinterpret_cast_59 = memref.reinterpret_cast %alloc_58 to offset: [0], sizes: [1, 128, 28, 28], strides: [100352, 784, 28, 1] : memref<1x128x784xf32> to memref<1x128x28x28xf32>
    %alloc_60 = memref.alloc() {alignment = 16 : i64} : memref<1x128x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %reinterpret_cast_59[%arg1, %arg2, %arg3, %arg4] : memref<1x128x28x28xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_60[%arg1, %arg2, %arg3, %arg4] : memref<1x128x28x28xf32>
          }
        }
      }
    }
    %alloc_61 = memref.alloc() {alignment = 16 : i64} : memref<1x128x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 128 {
          %124 = affine.apply #map15(%arg2, %arg3)
          affine.for %arg4 = 0 to 28 {
            affine.for %arg5 = 0 to 28 {
              %125 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map9(%arg4) to min #map29(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map11(%arg4, %arg5) to min #map30(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map20(%arg6, %arg2)
                    %131 = affine.apply #map14(%arg8, %arg4)
                    %132 = affine.apply #map14(%arg10, %arg5)
                    %133 = affine.load %alloc_60[%arg1, %130, %131, %132] : memref<1x128x28x28xf32>
                    %134 = affine.load %96[%124, %arg6, %arg8, %arg10] : memref<128x128x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %95[%124] : memref<128xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_61[%arg1, %124, %arg4, %arg5] : memref<1x128x28x28xf32>
            }
          }
        }
      }
    }
    %alloc_62 = memref.alloc() {alignment = 16 : i64} : memref<1x128x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %alloc_61[%arg1, %arg2, %arg3, %arg4] : memref<1x128x28x28xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_62[%arg1, %arg2, %arg3, %arg4] : memref<1x128x28x28xf32>
          }
        }
      }
    }
    %reinterpret_cast_63 = memref.reinterpret_cast %alloc_62 to offset: [0], sizes: [1, 128, 784], strides: [100352, 784, 1] : memref<1x128x28x28xf32> to memref<1x128x784xf32>
    %alloc_64 = memref.alloc() {alignment = 16 : i64} : memref<1x512x784xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 784 {
          %124 = affine.for %arg4 = 0 to 128 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %33[%arg2, %arg4] : memref<512x128xf32>
            %126 = affine.load %reinterpret_cast_63[%arg1, %arg4, %arg3] : memref<1x128x784xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_64[%arg1, %arg2, %arg3] : memref<1x512x784xf32>
        }
      }
    }
    %reinterpret_cast_65 = memref.reinterpret_cast %alloc_64 to offset: [0], sizes: [1, 512, 28, 28], strides: [401408, 784, 28, 1] : memref<1x512x784xf32> to memref<1x512x28x28xf32>
    %alloc_66 = memref.alloc() {alignment = 16 : i64} : memref<1x512x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %reinterpret_cast_65[%c0, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
            %125 = affine.load %alloc_54[%c0, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
            %126 = arith.addf %124, %125 : f32
            affine.store %126, %alloc_66[%arg1, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
          }
        }
      }
    }
    %alloc_67 = memref.alloc() {alignment = 16 : i64} : memref<1x512x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %alloc_66[%c0, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
            %125 = affine.load %94[%arg2, %c0, %c0] : memref<512x1x1xf32>
            %126 = arith.mulf %124, %125 : f32
            %127 = affine.load %93[%arg2, %c0, %c0] : memref<512x1x1xf32>
            %128 = arith.addf %126, %127 : f32
            %129 = arith.maxf %128, %cst_1 : f32
            affine.store %129, %alloc_67[%arg1, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
          }
        }
      }
    }
    %reinterpret_cast_68 = memref.reinterpret_cast %alloc_67 to offset: [0], sizes: [1, 512, 784], strides: [401408, 784, 1] : memref<1x512x28x28xf32> to memref<1x512x784xf32>
    %alloc_69 = memref.alloc() {alignment = 16 : i64} : memref<1x128x784xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 784 {
          %124 = affine.for %arg4 = 0 to 512 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %32[%arg2, %arg4] : memref<128x512xf32>
            %126 = affine.load %reinterpret_cast_68[%arg1, %arg4, %arg3] : memref<1x512x784xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_69[%arg1, %arg2, %arg3] : memref<1x128x784xf32>
        }
      }
    }
    %alloc_70 = memref.alloc() {alignment = 16 : i64} : memref<1x128x784xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 784 {
          %124 = affine.load %alloc_69[%c0, %arg2, %arg3] : memref<1x128x784xf32>
          %125 = affine.load %31[%c0, %arg2, %c0] : memref<1x128x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_70[%arg1, %arg2, %arg3] : memref<1x128x784xf32>
        }
      }
    }
    %reinterpret_cast_71 = memref.reinterpret_cast %alloc_70 to offset: [0], sizes: [1, 128, 28, 28], strides: [100352, 784, 28, 1] : memref<1x128x784xf32> to memref<1x128x28x28xf32>
    %alloc_72 = memref.alloc() {alignment = 16 : i64} : memref<1x128x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %reinterpret_cast_71[%arg1, %arg2, %arg3, %arg4] : memref<1x128x28x28xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_72[%arg1, %arg2, %arg3, %arg4] : memref<1x128x28x28xf32>
          }
        }
      }
    }
    %alloc_73 = memref.alloc() {alignment = 16 : i64} : memref<1x128x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 128 {
          %124 = affine.apply #map15(%arg2, %arg3)
          affine.for %arg4 = 0 to 28 {
            affine.for %arg5 = 0 to 28 {
              %125 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map9(%arg4) to min #map29(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map11(%arg4, %arg5) to min #map30(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map20(%arg6, %arg2)
                    %131 = affine.apply #map14(%arg8, %arg4)
                    %132 = affine.apply #map14(%arg10, %arg5)
                    %133 = affine.load %alloc_72[%arg1, %130, %131, %132] : memref<1x128x28x28xf32>
                    %134 = affine.load %92[%124, %arg6, %arg8, %arg10] : memref<128x128x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %91[%124] : memref<128xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_73[%arg1, %124, %arg4, %arg5] : memref<1x128x28x28xf32>
            }
          }
        }
      }
    }
    %alloc_74 = memref.alloc() {alignment = 16 : i64} : memref<1x128x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %alloc_73[%arg1, %arg2, %arg3, %arg4] : memref<1x128x28x28xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_74[%arg1, %arg2, %arg3, %arg4] : memref<1x128x28x28xf32>
          }
        }
      }
    }
    %reinterpret_cast_75 = memref.reinterpret_cast %alloc_74 to offset: [0], sizes: [1, 128, 784], strides: [100352, 784, 1] : memref<1x128x28x28xf32> to memref<1x128x784xf32>
    %alloc_76 = memref.alloc() {alignment = 16 : i64} : memref<1x512x784xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 784 {
          %124 = affine.for %arg4 = 0 to 128 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %30[%arg2, %arg4] : memref<512x128xf32>
            %126 = affine.load %reinterpret_cast_75[%arg1, %arg4, %arg3] : memref<1x128x784xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_76[%arg1, %arg2, %arg3] : memref<1x512x784xf32>
        }
      }
    }
    %reinterpret_cast_77 = memref.reinterpret_cast %alloc_76 to offset: [0], sizes: [1, 512, 28, 28], strides: [401408, 784, 28, 1] : memref<1x512x784xf32> to memref<1x512x28x28xf32>
    %alloc_78 = memref.alloc() {alignment = 16 : i64} : memref<1x512x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %reinterpret_cast_77[%c0, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
            %125 = affine.load %alloc_66[%c0, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
            %126 = arith.addf %124, %125 : f32
            affine.store %126, %alloc_78[%arg1, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
          }
        }
      }
    }
    %alloc_79 = memref.alloc() {alignment = 16 : i64} : memref<1x512x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %alloc_78[%c0, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
            %125 = affine.load %90[%arg2, %c0, %c0] : memref<512x1x1xf32>
            %126 = arith.mulf %124, %125 : f32
            %127 = affine.load %89[%arg2, %c0, %c0] : memref<512x1x1xf32>
            %128 = arith.addf %126, %127 : f32
            %129 = arith.maxf %128, %cst_1 : f32
            affine.store %129, %alloc_79[%arg1, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
          }
        }
      }
    }
    %reinterpret_cast_80 = memref.reinterpret_cast %alloc_79 to offset: [0], sizes: [1, 512, 784], strides: [401408, 784, 1] : memref<1x512x28x28xf32> to memref<1x512x784xf32>
    %alloc_81 = memref.alloc() {alignment = 16 : i64} : memref<1x128x784xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 784 {
          %124 = affine.for %arg4 = 0 to 512 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %29[%arg2, %arg4] : memref<128x512xf32>
            %126 = affine.load %reinterpret_cast_80[%arg1, %arg4, %arg3] : memref<1x512x784xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_81[%arg1, %arg2, %arg3] : memref<1x128x784xf32>
        }
      }
    }
    %alloc_82 = memref.alloc() {alignment = 16 : i64} : memref<1x128x784xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 784 {
          %124 = affine.load %alloc_81[%c0, %arg2, %arg3] : memref<1x128x784xf32>
          %125 = affine.load %28[%c0, %arg2, %c0] : memref<1x128x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_82[%arg1, %arg2, %arg3] : memref<1x128x784xf32>
        }
      }
    }
    %reinterpret_cast_83 = memref.reinterpret_cast %alloc_82 to offset: [0], sizes: [1, 128, 28, 28], strides: [100352, 784, 28, 1] : memref<1x128x784xf32> to memref<1x128x28x28xf32>
    %alloc_84 = memref.alloc() {alignment = 16 : i64} : memref<1x128x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %reinterpret_cast_83[%arg1, %arg2, %arg3, %arg4] : memref<1x128x28x28xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_84[%arg1, %arg2, %arg3, %arg4] : memref<1x128x28x28xf32>
          }
        }
      }
    }
    %alloc_85 = memref.alloc() {alignment = 16 : i64} : memref<1x128x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 128 {
          %124 = affine.apply #map15(%arg2, %arg3)
          affine.for %arg4 = 0 to 28 {
            affine.for %arg5 = 0 to 28 {
              %125 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map9(%arg4) to min #map29(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map11(%arg4, %arg5) to min #map30(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map20(%arg6, %arg2)
                    %131 = affine.apply #map14(%arg8, %arg4)
                    %132 = affine.apply #map14(%arg10, %arg5)
                    %133 = affine.load %alloc_84[%arg1, %130, %131, %132] : memref<1x128x28x28xf32>
                    %134 = affine.load %88[%124, %arg6, %arg8, %arg10] : memref<128x128x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %87[%124] : memref<128xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_85[%arg1, %124, %arg4, %arg5] : memref<1x128x28x28xf32>
            }
          }
        }
      }
    }
    %alloc_86 = memref.alloc() {alignment = 16 : i64} : memref<1x128x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %alloc_85[%arg1, %arg2, %arg3, %arg4] : memref<1x128x28x28xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_86[%arg1, %arg2, %arg3, %arg4] : memref<1x128x28x28xf32>
          }
        }
      }
    }
    %reinterpret_cast_87 = memref.reinterpret_cast %alloc_86 to offset: [0], sizes: [1, 128, 784], strides: [100352, 784, 1] : memref<1x128x28x28xf32> to memref<1x128x784xf32>
    %alloc_88 = memref.alloc() {alignment = 16 : i64} : memref<1x512x784xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 784 {
          %124 = affine.for %arg4 = 0 to 128 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %27[%arg2, %arg4] : memref<512x128xf32>
            %126 = affine.load %reinterpret_cast_87[%arg1, %arg4, %arg3] : memref<1x128x784xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_88[%arg1, %arg2, %arg3] : memref<1x512x784xf32>
        }
      }
    }
    %reinterpret_cast_89 = memref.reinterpret_cast %alloc_88 to offset: [0], sizes: [1, 512, 28, 28], strides: [401408, 784, 28, 1] : memref<1x512x784xf32> to memref<1x512x28x28xf32>
    %alloc_90 = memref.alloc() {alignment = 16 : i64} : memref<1x512x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %reinterpret_cast_89[%c0, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
            %125 = affine.load %alloc_78[%c0, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
            %126 = arith.addf %124, %125 : f32
            %127 = affine.load %86[%arg2, %c0, %c0] : memref<512x1x1xf32>
            %128 = arith.mulf %126, %127 : f32
            %129 = affine.load %85[%arg2, %c0, %c0] : memref<512x1x1xf32>
            %130 = arith.addf %128, %129 : f32
            %131 = arith.maxf %130, %cst_1 : f32
            affine.store %131, %alloc_90[%arg1, %arg2, %arg3, %arg4] : memref<1x512x28x28xf32>
          }
        }
      }
    }
    %reinterpret_cast_91 = memref.reinterpret_cast %alloc_90 to offset: [0], sizes: [1, 512, 784], strides: [401408, 784, 1] : memref<1x512x28x28xf32> to memref<1x512x784xf32>
    %alloc_92 = memref.alloc() {alignment = 16 : i64} : memref<1x256x784xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 784 {
          %124 = affine.for %arg4 = 0 to 512 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %26[%arg2, %arg4] : memref<256x512xf32>
            %126 = affine.load %reinterpret_cast_91[%arg1, %arg4, %arg3] : memref<1x512x784xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_92[%arg1, %arg2, %arg3] : memref<1x256x784xf32>
        }
      }
    }
    %alloc_93 = memref.alloc() {alignment = 16 : i64} : memref<1x256x784xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 784 {
          %124 = affine.load %alloc_92[%c0, %arg2, %arg3] : memref<1x256x784xf32>
          %125 = affine.load %25[%c0, %arg2, %c0] : memref<1x256x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_93[%arg1, %arg2, %arg3] : memref<1x256x784xf32>
        }
      }
    }
    %reinterpret_cast_94 = memref.reinterpret_cast %alloc_93 to offset: [0], sizes: [1, 256, 28, 28], strides: [200704, 784, 28, 1] : memref<1x256x784xf32> to memref<1x256x28x28xf32>
    %alloc_95 = memref.alloc() {alignment = 16 : i64} : memref<1x256x28x28xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %124 = affine.load %reinterpret_cast_94[%arg1, %arg2, %arg3, %arg4] : memref<1x256x28x28xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_95[%arg1, %arg2, %arg3, %arg4] : memref<1x256x28x28xf32>
          }
        }
      }
    }
    %alloc_96 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 256 {
          %124 = affine.apply #map31(%arg2, %arg3)
          affine.for %arg4 = 0 to 14 {
            affine.for %arg5 = 0 to 14 {
              %125 = affine.for %arg6 = 0 to 256 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map16(%arg4) to min #map32(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map18(%arg4, %arg5) to min #map33(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map27(%arg6, %arg2)
                    %131 = affine.apply #map21(%arg8, %arg4)
                    %132 = affine.apply #map21(%arg10, %arg5)
                    %133 = affine.load %alloc_95[%arg1, %130, %131, %132] : memref<1x256x28x28xf32>
                    %134 = affine.load %84[%124, %arg6, %arg8, %arg10] : memref<256x256x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %83[%124] : memref<256xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_96[%arg1, %124, %arg4, %arg5] : memref<1x256x14x14xf32>
            }
          }
        }
      }
    }
    %alloc_97 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %alloc_96[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_97[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
          }
        }
      }
    }
    %reinterpret_cast_98 = memref.reinterpret_cast %alloc_97 to offset: [0], sizes: [1, 256, 196], strides: [50176, 196, 1] : memref<1x256x14x14xf32> to memref<1x256x196xf32>
    %alloc_99 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.for %arg4 = 0 to 256 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %24[%arg2, %arg4] : memref<1024x256xf32>
            %126 = affine.load %reinterpret_cast_98[%arg1, %arg4, %arg3] : memref<1x256x196xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_99[%arg1, %arg2, %arg3] : memref<1x1024x196xf32>
        }
      }
    }
    %reinterpret_cast_100 = memref.reinterpret_cast %alloc_99 to offset: [0], sizes: [1, 1024, 14, 14], strides: [200704, 196, 14, 1] : memref<1x1024x196xf32> to memref<1x1024x14x14xf32>
    %alloc_101 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 1024 {
          %124 = affine.apply #map34(%arg2, %arg3)
          affine.for %arg4 = 0 to 14 {
            affine.for %arg5 = 0 to 14 {
              %125 = affine.for %arg6 = 0 to 512 iter_args(%arg7 = %cst_1) -> (f32) {
                %126 = affine.for %arg8 = max #map23(%arg4) to min #map35(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %127 = affine.for %arg10 = max #map25(%arg4, %arg5) to min #map36(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %128 = affine.apply #map37(%arg6, %arg2)
                    %129 = affine.apply #map28(%arg8, %arg4)
                    %130 = affine.apply #map28(%arg10, %arg5)
                    %131 = affine.load %alloc_90[%arg1, %128, %129, %130] : memref<1x512x28x28xf32>
                    %132 = affine.load %119[%124, %arg6, %arg8, %arg10] : memref<1024x512x1x1xf32>
                    %133 = arith.mulf %131, %132 : f32
                    %134 = arith.addf %arg11, %133 : f32
                    affine.yield %134 : f32
                  }
                  affine.yield %127 : f32
                }
                affine.yield %126 : f32
              }
              affine.store %125, %alloc_101[%arg1, %124, %arg4, %arg5] : memref<1x1024x14x14xf32>
            }
          }
        }
      }
    }
    %alloc_102 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %reinterpret_cast_100[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %125 = affine.load %alloc_101[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %126 = arith.addf %124, %125 : f32
            affine.store %126, %alloc_102[%arg1, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
          }
        }
      }
    }
    %alloc_103 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %alloc_102[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %125 = affine.load %82[%arg2, %c0, %c0] : memref<1024x1x1xf32>
            %126 = arith.mulf %124, %125 : f32
            %127 = affine.load %81[%arg2, %c0, %c0] : memref<1024x1x1xf32>
            %128 = arith.addf %126, %127 : f32
            %129 = arith.maxf %128, %cst_1 : f32
            affine.store %129, %alloc_103[%arg1, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
          }
        }
      }
    }
    %reinterpret_cast_104 = memref.reinterpret_cast %alloc_103 to offset: [0], sizes: [1, 1024, 196], strides: [200704, 196, 1] : memref<1x1024x14x14xf32> to memref<1x1024x196xf32>
    %alloc_105 = memref.alloc() {alignment = 16 : i64} : memref<1x256x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.for %arg4 = 0 to 1024 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %23[%arg2, %arg4] : memref<256x1024xf32>
            %126 = affine.load %reinterpret_cast_104[%arg1, %arg4, %arg3] : memref<1x1024x196xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_105[%arg1, %arg2, %arg3] : memref<1x256x196xf32>
        }
      }
    }
    %alloc_106 = memref.alloc() {alignment = 16 : i64} : memref<1x256x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.load %alloc_105[%c0, %arg2, %arg3] : memref<1x256x196xf32>
          %125 = affine.load %22[%c0, %arg2, %c0] : memref<1x256x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_106[%arg1, %arg2, %arg3] : memref<1x256x196xf32>
        }
      }
    }
    %reinterpret_cast_107 = memref.reinterpret_cast %alloc_106 to offset: [0], sizes: [1, 256, 14, 14], strides: [50176, 196, 14, 1] : memref<1x256x196xf32> to memref<1x256x14x14xf32>
    %alloc_108 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %reinterpret_cast_107[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_108[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
          }
        }
      }
    }
    %alloc_109 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 256 {
          %124 = affine.apply #map31(%arg2, %arg3)
          affine.for %arg4 = 0 to 14 {
            affine.for %arg5 = 0 to 14 {
              %125 = affine.for %arg6 = 0 to 256 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map9(%arg4) to min #map38(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map11(%arg4, %arg5) to min #map39(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map27(%arg6, %arg2)
                    %131 = affine.apply #map14(%arg8, %arg4)
                    %132 = affine.apply #map14(%arg10, %arg5)
                    %133 = affine.load %alloc_108[%arg1, %130, %131, %132] : memref<1x256x14x14xf32>
                    %134 = affine.load %80[%124, %arg6, %arg8, %arg10] : memref<256x256x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %79[%124] : memref<256xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_109[%arg1, %124, %arg4, %arg5] : memref<1x256x14x14xf32>
            }
          }
        }
      }
    }
    %alloc_110 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %alloc_109[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_110[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
          }
        }
      }
    }
    %reinterpret_cast_111 = memref.reinterpret_cast %alloc_110 to offset: [0], sizes: [1, 256, 196], strides: [50176, 196, 1] : memref<1x256x14x14xf32> to memref<1x256x196xf32>
    %alloc_112 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.for %arg4 = 0 to 256 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %21[%arg2, %arg4] : memref<1024x256xf32>
            %126 = affine.load %reinterpret_cast_111[%arg1, %arg4, %arg3] : memref<1x256x196xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_112[%arg1, %arg2, %arg3] : memref<1x1024x196xf32>
        }
      }
    }
    %reinterpret_cast_113 = memref.reinterpret_cast %alloc_112 to offset: [0], sizes: [1, 1024, 14, 14], strides: [200704, 196, 14, 1] : memref<1x1024x196xf32> to memref<1x1024x14x14xf32>
    %alloc_114 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %reinterpret_cast_113[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %125 = affine.load %alloc_102[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %126 = arith.addf %124, %125 : f32
            affine.store %126, %alloc_114[%arg1, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
          }
        }
      }
    }
    %alloc_115 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %alloc_114[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %125 = affine.load %78[%arg2, %c0, %c0] : memref<1024x1x1xf32>
            %126 = arith.mulf %124, %125 : f32
            %127 = affine.load %77[%arg2, %c0, %c0] : memref<1024x1x1xf32>
            %128 = arith.addf %126, %127 : f32
            %129 = arith.maxf %128, %cst_1 : f32
            affine.store %129, %alloc_115[%arg1, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
          }
        }
      }
    }
    %reinterpret_cast_116 = memref.reinterpret_cast %alloc_115 to offset: [0], sizes: [1, 1024, 196], strides: [200704, 196, 1] : memref<1x1024x14x14xf32> to memref<1x1024x196xf32>
    %alloc_117 = memref.alloc() {alignment = 16 : i64} : memref<1x256x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.for %arg4 = 0 to 1024 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %20[%arg2, %arg4] : memref<256x1024xf32>
            %126 = affine.load %reinterpret_cast_116[%arg1, %arg4, %arg3] : memref<1x1024x196xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_117[%arg1, %arg2, %arg3] : memref<1x256x196xf32>
        }
      }
    }
    %alloc_118 = memref.alloc() {alignment = 16 : i64} : memref<1x256x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.load %alloc_117[%c0, %arg2, %arg3] : memref<1x256x196xf32>
          %125 = affine.load %19[%c0, %arg2, %c0] : memref<1x256x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_118[%arg1, %arg2, %arg3] : memref<1x256x196xf32>
        }
      }
    }
    %reinterpret_cast_119 = memref.reinterpret_cast %alloc_118 to offset: [0], sizes: [1, 256, 14, 14], strides: [50176, 196, 14, 1] : memref<1x256x196xf32> to memref<1x256x14x14xf32>
    %alloc_120 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %reinterpret_cast_119[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_120[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
          }
        }
      }
    }
    %alloc_121 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 256 {
          %124 = affine.apply #map31(%arg2, %arg3)
          affine.for %arg4 = 0 to 14 {
            affine.for %arg5 = 0 to 14 {
              %125 = affine.for %arg6 = 0 to 256 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map9(%arg4) to min #map38(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map11(%arg4, %arg5) to min #map39(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map27(%arg6, %arg2)
                    %131 = affine.apply #map14(%arg8, %arg4)
                    %132 = affine.apply #map14(%arg10, %arg5)
                    %133 = affine.load %alloc_120[%arg1, %130, %131, %132] : memref<1x256x14x14xf32>
                    %134 = affine.load %76[%124, %arg6, %arg8, %arg10] : memref<256x256x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %75[%124] : memref<256xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_121[%arg1, %124, %arg4, %arg5] : memref<1x256x14x14xf32>
            }
          }
        }
      }
    }
    %alloc_122 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %alloc_121[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_122[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
          }
        }
      }
    }
    %reinterpret_cast_123 = memref.reinterpret_cast %alloc_122 to offset: [0], sizes: [1, 256, 196], strides: [50176, 196, 1] : memref<1x256x14x14xf32> to memref<1x256x196xf32>
    %alloc_124 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.for %arg4 = 0 to 256 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %18[%arg2, %arg4] : memref<1024x256xf32>
            %126 = affine.load %reinterpret_cast_123[%arg1, %arg4, %arg3] : memref<1x256x196xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_124[%arg1, %arg2, %arg3] : memref<1x1024x196xf32>
        }
      }
    }
    %reinterpret_cast_125 = memref.reinterpret_cast %alloc_124 to offset: [0], sizes: [1, 1024, 14, 14], strides: [200704, 196, 14, 1] : memref<1x1024x196xf32> to memref<1x1024x14x14xf32>
    %alloc_126 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %reinterpret_cast_125[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %125 = affine.load %alloc_114[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %126 = arith.addf %124, %125 : f32
            affine.store %126, %alloc_126[%arg1, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
          }
        }
      }
    }
    %alloc_127 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %alloc_126[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %125 = affine.load %74[%arg2, %c0, %c0] : memref<1024x1x1xf32>
            %126 = arith.mulf %124, %125 : f32
            %127 = affine.load %73[%arg2, %c0, %c0] : memref<1024x1x1xf32>
            %128 = arith.addf %126, %127 : f32
            %129 = arith.maxf %128, %cst_1 : f32
            affine.store %129, %alloc_127[%arg1, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
          }
        }
      }
    }
    %reinterpret_cast_128 = memref.reinterpret_cast %alloc_127 to offset: [0], sizes: [1, 1024, 196], strides: [200704, 196, 1] : memref<1x1024x14x14xf32> to memref<1x1024x196xf32>
    %alloc_129 = memref.alloc() {alignment = 16 : i64} : memref<1x256x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.for %arg4 = 0 to 1024 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %17[%arg2, %arg4] : memref<256x1024xf32>
            %126 = affine.load %reinterpret_cast_128[%arg1, %arg4, %arg3] : memref<1x1024x196xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_129[%arg1, %arg2, %arg3] : memref<1x256x196xf32>
        }
      }
    }
    %alloc_130 = memref.alloc() {alignment = 16 : i64} : memref<1x256x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.load %alloc_129[%c0, %arg2, %arg3] : memref<1x256x196xf32>
          %125 = affine.load %16[%c0, %arg2, %c0] : memref<1x256x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_130[%arg1, %arg2, %arg3] : memref<1x256x196xf32>
        }
      }
    }
    %reinterpret_cast_131 = memref.reinterpret_cast %alloc_130 to offset: [0], sizes: [1, 256, 14, 14], strides: [50176, 196, 14, 1] : memref<1x256x196xf32> to memref<1x256x14x14xf32>
    %alloc_132 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %reinterpret_cast_131[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_132[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
          }
        }
      }
    }
    %alloc_133 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 256 {
          %124 = affine.apply #map31(%arg2, %arg3)
          affine.for %arg4 = 0 to 14 {
            affine.for %arg5 = 0 to 14 {
              %125 = affine.for %arg6 = 0 to 256 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map9(%arg4) to min #map38(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map11(%arg4, %arg5) to min #map39(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map27(%arg6, %arg2)
                    %131 = affine.apply #map14(%arg8, %arg4)
                    %132 = affine.apply #map14(%arg10, %arg5)
                    %133 = affine.load %alloc_132[%arg1, %130, %131, %132] : memref<1x256x14x14xf32>
                    %134 = affine.load %72[%124, %arg6, %arg8, %arg10] : memref<256x256x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %71[%124] : memref<256xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_133[%arg1, %124, %arg4, %arg5] : memref<1x256x14x14xf32>
            }
          }
        }
      }
    }
    %alloc_134 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %alloc_133[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_134[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
          }
        }
      }
    }
    %reinterpret_cast_135 = memref.reinterpret_cast %alloc_134 to offset: [0], sizes: [1, 256, 196], strides: [50176, 196, 1] : memref<1x256x14x14xf32> to memref<1x256x196xf32>
    %alloc_136 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.for %arg4 = 0 to 256 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %15[%arg2, %arg4] : memref<1024x256xf32>
            %126 = affine.load %reinterpret_cast_135[%arg1, %arg4, %arg3] : memref<1x256x196xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_136[%arg1, %arg2, %arg3] : memref<1x1024x196xf32>
        }
      }
    }
    %reinterpret_cast_137 = memref.reinterpret_cast %alloc_136 to offset: [0], sizes: [1, 1024, 14, 14], strides: [200704, 196, 14, 1] : memref<1x1024x196xf32> to memref<1x1024x14x14xf32>
    %alloc_138 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %reinterpret_cast_137[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %125 = affine.load %alloc_126[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %126 = arith.addf %124, %125 : f32
            affine.store %126, %alloc_138[%arg1, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
          }
        }
      }
    }
    %alloc_139 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %alloc_138[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %125 = affine.load %70[%arg2, %c0, %c0] : memref<1024x1x1xf32>
            %126 = arith.mulf %124, %125 : f32
            %127 = affine.load %69[%arg2, %c0, %c0] : memref<1024x1x1xf32>
            %128 = arith.addf %126, %127 : f32
            %129 = arith.maxf %128, %cst_1 : f32
            affine.store %129, %alloc_139[%arg1, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
          }
        }
      }
    }
    %reinterpret_cast_140 = memref.reinterpret_cast %alloc_139 to offset: [0], sizes: [1, 1024, 196], strides: [200704, 196, 1] : memref<1x1024x14x14xf32> to memref<1x1024x196xf32>
    %alloc_141 = memref.alloc() {alignment = 16 : i64} : memref<1x256x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.for %arg4 = 0 to 1024 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %14[%arg2, %arg4] : memref<256x1024xf32>
            %126 = affine.load %reinterpret_cast_140[%arg1, %arg4, %arg3] : memref<1x1024x196xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_141[%arg1, %arg2, %arg3] : memref<1x256x196xf32>
        }
      }
    }
    %alloc_142 = memref.alloc() {alignment = 16 : i64} : memref<1x256x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.load %alloc_141[%c0, %arg2, %arg3] : memref<1x256x196xf32>
          %125 = affine.load %13[%c0, %arg2, %c0] : memref<1x256x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_142[%arg1, %arg2, %arg3] : memref<1x256x196xf32>
        }
      }
    }
    %reinterpret_cast_143 = memref.reinterpret_cast %alloc_142 to offset: [0], sizes: [1, 256, 14, 14], strides: [50176, 196, 14, 1] : memref<1x256x196xf32> to memref<1x256x14x14xf32>
    %alloc_144 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %reinterpret_cast_143[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_144[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
          }
        }
      }
    }
    %alloc_145 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 256 {
          %124 = affine.apply #map31(%arg2, %arg3)
          affine.for %arg4 = 0 to 14 {
            affine.for %arg5 = 0 to 14 {
              %125 = affine.for %arg6 = 0 to 256 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map9(%arg4) to min #map38(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map11(%arg4, %arg5) to min #map39(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map27(%arg6, %arg2)
                    %131 = affine.apply #map14(%arg8, %arg4)
                    %132 = affine.apply #map14(%arg10, %arg5)
                    %133 = affine.load %alloc_144[%arg1, %130, %131, %132] : memref<1x256x14x14xf32>
                    %134 = affine.load %68[%124, %arg6, %arg8, %arg10] : memref<256x256x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %67[%124] : memref<256xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_145[%arg1, %124, %arg4, %arg5] : memref<1x256x14x14xf32>
            }
          }
        }
      }
    }
    %alloc_146 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %alloc_145[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_146[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
          }
        }
      }
    }
    %reinterpret_cast_147 = memref.reinterpret_cast %alloc_146 to offset: [0], sizes: [1, 256, 196], strides: [50176, 196, 1] : memref<1x256x14x14xf32> to memref<1x256x196xf32>
    %alloc_148 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.for %arg4 = 0 to 256 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %12[%arg2, %arg4] : memref<1024x256xf32>
            %126 = affine.load %reinterpret_cast_147[%arg1, %arg4, %arg3] : memref<1x256x196xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_148[%arg1, %arg2, %arg3] : memref<1x1024x196xf32>
        }
      }
    }
    %reinterpret_cast_149 = memref.reinterpret_cast %alloc_148 to offset: [0], sizes: [1, 1024, 14, 14], strides: [200704, 196, 14, 1] : memref<1x1024x196xf32> to memref<1x1024x14x14xf32>
    %alloc_150 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %reinterpret_cast_149[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %125 = affine.load %alloc_138[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %126 = arith.addf %124, %125 : f32
            affine.store %126, %alloc_150[%arg1, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
          }
        }
      }
    }
    %alloc_151 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %alloc_150[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %125 = affine.load %66[%arg2, %c0, %c0] : memref<1024x1x1xf32>
            %126 = arith.mulf %124, %125 : f32
            %127 = affine.load %65[%arg2, %c0, %c0] : memref<1024x1x1xf32>
            %128 = arith.addf %126, %127 : f32
            %129 = arith.maxf %128, %cst_1 : f32
            affine.store %129, %alloc_151[%arg1, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
          }
        }
      }
    }
    %reinterpret_cast_152 = memref.reinterpret_cast %alloc_151 to offset: [0], sizes: [1, 1024, 196], strides: [200704, 196, 1] : memref<1x1024x14x14xf32> to memref<1x1024x196xf32>
    %alloc_153 = memref.alloc() {alignment = 16 : i64} : memref<1x256x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.for %arg4 = 0 to 1024 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %11[%arg2, %arg4] : memref<256x1024xf32>
            %126 = affine.load %reinterpret_cast_152[%arg1, %arg4, %arg3] : memref<1x1024x196xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_153[%arg1, %arg2, %arg3] : memref<1x256x196xf32>
        }
      }
    }
    %alloc_154 = memref.alloc() {alignment = 16 : i64} : memref<1x256x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.load %alloc_153[%c0, %arg2, %arg3] : memref<1x256x196xf32>
          %125 = affine.load %10[%c0, %arg2, %c0] : memref<1x256x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_154[%arg1, %arg2, %arg3] : memref<1x256x196xf32>
        }
      }
    }
    %reinterpret_cast_155 = memref.reinterpret_cast %alloc_154 to offset: [0], sizes: [1, 256, 14, 14], strides: [50176, 196, 14, 1] : memref<1x256x196xf32> to memref<1x256x14x14xf32>
    %alloc_156 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %reinterpret_cast_155[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_156[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
          }
        }
      }
    }
    %alloc_157 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 256 {
          %124 = affine.apply #map31(%arg2, %arg3)
          affine.for %arg4 = 0 to 14 {
            affine.for %arg5 = 0 to 14 {
              %125 = affine.for %arg6 = 0 to 256 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map9(%arg4) to min #map38(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map11(%arg4, %arg5) to min #map39(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map27(%arg6, %arg2)
                    %131 = affine.apply #map14(%arg8, %arg4)
                    %132 = affine.apply #map14(%arg10, %arg5)
                    %133 = affine.load %alloc_156[%arg1, %130, %131, %132] : memref<1x256x14x14xf32>
                    %134 = affine.load %64[%124, %arg6, %arg8, %arg10] : memref<256x256x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %63[%124] : memref<256xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_157[%arg1, %124, %arg4, %arg5] : memref<1x256x14x14xf32>
            }
          }
        }
      }
    }
    %alloc_158 = memref.alloc() {alignment = 16 : i64} : memref<1x256x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 256 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %alloc_157[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_158[%arg1, %arg2, %arg3, %arg4] : memref<1x256x14x14xf32>
          }
        }
      }
    }
    %reinterpret_cast_159 = memref.reinterpret_cast %alloc_158 to offset: [0], sizes: [1, 256, 196], strides: [50176, 196, 1] : memref<1x256x14x14xf32> to memref<1x256x196xf32>
    %alloc_160 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.for %arg4 = 0 to 256 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %9[%arg2, %arg4] : memref<1024x256xf32>
            %126 = affine.load %reinterpret_cast_159[%arg1, %arg4, %arg3] : memref<1x256x196xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_160[%arg1, %arg2, %arg3] : memref<1x1024x196xf32>
        }
      }
    }
    %reinterpret_cast_161 = memref.reinterpret_cast %alloc_160 to offset: [0], sizes: [1, 1024, 14, 14], strides: [200704, 196, 14, 1] : memref<1x1024x196xf32> to memref<1x1024x14x14xf32>
    %alloc_162 = memref.alloc() {alignment = 16 : i64} : memref<1x1024x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1024 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %reinterpret_cast_161[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %125 = affine.load %alloc_150[%c0, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
            %126 = arith.addf %124, %125 : f32
            %127 = affine.load %62[%arg2, %c0, %c0] : memref<1024x1x1xf32>
            %128 = arith.mulf %126, %127 : f32
            %129 = affine.load %61[%arg2, %c0, %c0] : memref<1024x1x1xf32>
            %130 = arith.addf %128, %129 : f32
            %131 = arith.maxf %130, %cst_1 : f32
            affine.store %131, %alloc_162[%arg1, %arg2, %arg3, %arg4] : memref<1x1024x14x14xf32>
          }
        }
      }
    }
    %reinterpret_cast_163 = memref.reinterpret_cast %alloc_162 to offset: [0], sizes: [1, 1024, 196], strides: [200704, 196, 1] : memref<1x1024x14x14xf32> to memref<1x1024x196xf32>
    %alloc_164 = memref.alloc() {alignment = 16 : i64} : memref<1x512x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.for %arg4 = 0 to 1024 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %8[%arg2, %arg4] : memref<512x1024xf32>
            %126 = affine.load %reinterpret_cast_163[%arg1, %arg4, %arg3] : memref<1x1024x196xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_164[%arg1, %arg2, %arg3] : memref<1x512x196xf32>
        }
      }
    }
    %alloc_165 = memref.alloc() {alignment = 16 : i64} : memref<1x512x196xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 196 {
          %124 = affine.load %alloc_164[%c0, %arg2, %arg3] : memref<1x512x196xf32>
          %125 = affine.load %7[%c0, %arg2, %c0] : memref<1x512x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_165[%arg1, %arg2, %arg3] : memref<1x512x196xf32>
        }
      }
    }
    %reinterpret_cast_166 = memref.reinterpret_cast %alloc_165 to offset: [0], sizes: [1, 512, 14, 14], strides: [100352, 196, 14, 1] : memref<1x512x196xf32> to memref<1x512x14x14xf32>
    %alloc_167 = memref.alloc() {alignment = 16 : i64} : memref<1x512x14x14xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 14 {
          affine.for %arg4 = 0 to 14 {
            %124 = affine.load %reinterpret_cast_166[%arg1, %arg2, %arg3, %arg4] : memref<1x512x14x14xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_167[%arg1, %arg2, %arg3, %arg4] : memref<1x512x14x14xf32>
          }
        }
      }
    }
    %alloc_168 = memref.alloc() {alignment = 16 : i64} : memref<1x512x7x7xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 512 {
          %124 = affine.apply #map22(%arg2, %arg3)
          affine.for %arg4 = 0 to 7 {
            affine.for %arg5 = 0 to 7 {
              %125 = affine.for %arg6 = 0 to 512 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map16(%arg4) to min #map40(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map18(%arg4, %arg5) to min #map41(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map37(%arg6, %arg2)
                    %131 = affine.apply #map21(%arg8, %arg4)
                    %132 = affine.apply #map21(%arg10, %arg5)
                    %133 = affine.load %alloc_167[%arg1, %130, %131, %132] : memref<1x512x14x14xf32>
                    %134 = affine.load %60[%124, %arg6, %arg8, %arg10] : memref<512x512x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %59[%124] : memref<512xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_168[%arg1, %124, %arg4, %arg5] : memref<1x512x7x7xf32>
            }
          }
        }
      }
    }
    %alloc_169 = memref.alloc() {alignment = 16 : i64} : memref<1x512x7x7xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 7 {
          affine.for %arg4 = 0 to 7 {
            %124 = affine.load %alloc_168[%arg1, %arg2, %arg3, %arg4] : memref<1x512x7x7xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_169[%arg1, %arg2, %arg3, %arg4] : memref<1x512x7x7xf32>
          }
        }
      }
    }
    %reinterpret_cast_170 = memref.reinterpret_cast %alloc_169 to offset: [0], sizes: [1, 512, 49], strides: [25088, 49, 1] : memref<1x512x7x7xf32> to memref<1x512x49xf32>
    %alloc_171 = memref.alloc() {alignment = 16 : i64} : memref<1x2048x49xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 2048 {
        affine.for %arg3 = 0 to 49 {
          %124 = affine.for %arg4 = 0 to 512 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %6[%arg2, %arg4] : memref<2048x512xf32>
            %126 = affine.load %reinterpret_cast_170[%arg1, %arg4, %arg3] : memref<1x512x49xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_171[%arg1, %arg2, %arg3] : memref<1x2048x49xf32>
        }
      }
    }
    %reinterpret_cast_172 = memref.reinterpret_cast %alloc_171 to offset: [0], sizes: [1, 2048, 7, 7], strides: [100352, 49, 7, 1] : memref<1x2048x49xf32> to memref<1x2048x7x7xf32>
    %alloc_173 = memref.alloc() {alignment = 16 : i64} : memref<1x2048x7x7xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 2048 {
          %124 = affine.apply #map42(%arg2, %arg3)
          affine.for %arg4 = 0 to 7 {
            affine.for %arg5 = 0 to 7 {
              %125 = affine.for %arg6 = 0 to 1024 iter_args(%arg7 = %cst_1) -> (f32) {
                %126 = affine.for %arg8 = max #map23(%arg4) to min #map43(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %127 = affine.for %arg10 = max #map25(%arg4, %arg5) to min #map44(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %128 = affine.apply #map45(%arg6, %arg2)
                    %129 = affine.apply #map28(%arg8, %arg4)
                    %130 = affine.apply #map28(%arg10, %arg5)
                    %131 = affine.load %alloc_162[%arg1, %128, %129, %130] : memref<1x1024x14x14xf32>
                    %132 = affine.load %120[%124, %arg6, %arg8, %arg10] : memref<2048x1024x1x1xf32>
                    %133 = arith.mulf %131, %132 : f32
                    %134 = arith.addf %arg11, %133 : f32
                    affine.yield %134 : f32
                  }
                  affine.yield %127 : f32
                }
                affine.yield %126 : f32
              }
              affine.store %125, %alloc_173[%arg1, %124, %arg4, %arg5] : memref<1x2048x7x7xf32>
            }
          }
        }
      }
    }
    %alloc_174 = memref.alloc() {alignment = 16 : i64} : memref<1x2048x7x7xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 2048 {
        affine.for %arg3 = 0 to 7 {
          affine.for %arg4 = 0 to 7 {
            %124 = affine.load %reinterpret_cast_172[%c0, %arg2, %arg3, %arg4] : memref<1x2048x7x7xf32>
            %125 = affine.load %alloc_173[%c0, %arg2, %arg3, %arg4] : memref<1x2048x7x7xf32>
            %126 = arith.addf %124, %125 : f32
            affine.store %126, %alloc_174[%arg1, %arg2, %arg3, %arg4] : memref<1x2048x7x7xf32>
          }
        }
      }
    }
    %alloc_175 = memref.alloc() {alignment = 16 : i64} : memref<1x2048x7x7xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 2048 {
        affine.for %arg3 = 0 to 7 {
          affine.for %arg4 = 0 to 7 {
            %124 = affine.load %alloc_174[%c0, %arg2, %arg3, %arg4] : memref<1x2048x7x7xf32>
            %125 = affine.load %58[%arg2, %c0, %c0] : memref<2048x1x1xf32>
            %126 = arith.mulf %124, %125 : f32
            %127 = affine.load %57[%arg2, %c0, %c0] : memref<2048x1x1xf32>
            %128 = arith.addf %126, %127 : f32
            %129 = arith.maxf %128, %cst_1 : f32
            affine.store %129, %alloc_175[%arg1, %arg2, %arg3, %arg4] : memref<1x2048x7x7xf32>
          }
        }
      }
    }
    %reinterpret_cast_176 = memref.reinterpret_cast %alloc_175 to offset: [0], sizes: [1, 2048, 49], strides: [100352, 49, 1] : memref<1x2048x7x7xf32> to memref<1x2048x49xf32>
    %alloc_177 = memref.alloc() {alignment = 16 : i64} : memref<1x512x49xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 49 {
          %124 = affine.for %arg4 = 0 to 2048 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %5[%arg2, %arg4] : memref<512x2048xf32>
            %126 = affine.load %reinterpret_cast_176[%arg1, %arg4, %arg3] : memref<1x2048x49xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_177[%arg1, %arg2, %arg3] : memref<1x512x49xf32>
        }
      }
    }
    %alloc_178 = memref.alloc() {alignment = 16 : i64} : memref<1x512x49xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 49 {
          %124 = affine.load %alloc_177[%c0, %arg2, %arg3] : memref<1x512x49xf32>
          %125 = affine.load %4[%c0, %arg2, %c0] : memref<1x512x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_178[%arg1, %arg2, %arg3] : memref<1x512x49xf32>
        }
      }
    }
    %reinterpret_cast_179 = memref.reinterpret_cast %alloc_178 to offset: [0], sizes: [1, 512, 7, 7], strides: [25088, 49, 7, 1] : memref<1x512x49xf32> to memref<1x512x7x7xf32>
    %alloc_180 = memref.alloc() {alignment = 16 : i64} : memref<1x512x7x7xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 7 {
          affine.for %arg4 = 0 to 7 {
            %124 = affine.load %reinterpret_cast_179[%arg1, %arg2, %arg3, %arg4] : memref<1x512x7x7xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_180[%arg1, %arg2, %arg3, %arg4] : memref<1x512x7x7xf32>
          }
        }
      }
    }
    %alloc_181 = memref.alloc() {alignment = 16 : i64} : memref<1x512x7x7xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 512 {
          %124 = affine.apply #map22(%arg2, %arg3)
          affine.for %arg4 = 0 to 7 {
            affine.for %arg5 = 0 to 7 {
              %125 = affine.for %arg6 = 0 to 512 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map9(%arg4) to min #map46(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map11(%arg4, %arg5) to min #map47(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map37(%arg6, %arg2)
                    %131 = affine.apply #map14(%arg8, %arg4)
                    %132 = affine.apply #map14(%arg10, %arg5)
                    %133 = affine.load %alloc_180[%arg1, %130, %131, %132] : memref<1x512x7x7xf32>
                    %134 = affine.load %56[%124, %arg6, %arg8, %arg10] : memref<512x512x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %55[%124] : memref<512xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_181[%arg1, %124, %arg4, %arg5] : memref<1x512x7x7xf32>
            }
          }
        }
      }
    }
    %alloc_182 = memref.alloc() {alignment = 16 : i64} : memref<1x512x7x7xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 7 {
          affine.for %arg4 = 0 to 7 {
            %124 = affine.load %alloc_181[%arg1, %arg2, %arg3, %arg4] : memref<1x512x7x7xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_182[%arg1, %arg2, %arg3, %arg4] : memref<1x512x7x7xf32>
          }
        }
      }
    }
    %reinterpret_cast_183 = memref.reinterpret_cast %alloc_182 to offset: [0], sizes: [1, 512, 49], strides: [25088, 49, 1] : memref<1x512x7x7xf32> to memref<1x512x49xf32>
    %alloc_184 = memref.alloc() {alignment = 16 : i64} : memref<1x2048x49xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 2048 {
        affine.for %arg3 = 0 to 49 {
          %124 = affine.for %arg4 = 0 to 512 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %3[%arg2, %arg4] : memref<2048x512xf32>
            %126 = affine.load %reinterpret_cast_183[%arg1, %arg4, %arg3] : memref<1x512x49xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_184[%arg1, %arg2, %arg3] : memref<1x2048x49xf32>
        }
      }
    }
    %reinterpret_cast_185 = memref.reinterpret_cast %alloc_184 to offset: [0], sizes: [1, 2048, 7, 7], strides: [100352, 49, 7, 1] : memref<1x2048x49xf32> to memref<1x2048x7x7xf32>
    %alloc_186 = memref.alloc() {alignment = 16 : i64} : memref<1x2048x7x7xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 2048 {
        affine.for %arg3 = 0 to 7 {
          affine.for %arg4 = 0 to 7 {
            %124 = affine.load %reinterpret_cast_185[%c0, %arg2, %arg3, %arg4] : memref<1x2048x7x7xf32>
            %125 = affine.load %alloc_174[%c0, %arg2, %arg3, %arg4] : memref<1x2048x7x7xf32>
            %126 = arith.addf %124, %125 : f32
            affine.store %126, %alloc_186[%arg1, %arg2, %arg3, %arg4] : memref<1x2048x7x7xf32>
          }
        }
      }
    }
    %alloc_187 = memref.alloc() {alignment = 16 : i64} : memref<1x2048x7x7xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 2048 {
        affine.for %arg3 = 0 to 7 {
          affine.for %arg4 = 0 to 7 {
            %124 = affine.load %alloc_186[%c0, %arg2, %arg3, %arg4] : memref<1x2048x7x7xf32>
            %125 = affine.load %54[%arg2, %c0, %c0] : memref<2048x1x1xf32>
            %126 = arith.mulf %124, %125 : f32
            %127 = affine.load %53[%arg2, %c0, %c0] : memref<2048x1x1xf32>
            %128 = arith.addf %126, %127 : f32
            %129 = arith.maxf %128, %cst_1 : f32
            affine.store %129, %alloc_187[%arg1, %arg2, %arg3, %arg4] : memref<1x2048x7x7xf32>
          }
        }
      }
    }
    %reinterpret_cast_188 = memref.reinterpret_cast %alloc_187 to offset: [0], sizes: [1, 2048, 49], strides: [100352, 49, 1] : memref<1x2048x7x7xf32> to memref<1x2048x49xf32>
    %alloc_189 = memref.alloc() {alignment = 16 : i64} : memref<1x512x49xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 49 {
          %124 = affine.for %arg4 = 0 to 2048 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %2[%arg2, %arg4] : memref<512x2048xf32>
            %126 = affine.load %reinterpret_cast_188[%arg1, %arg4, %arg3] : memref<1x2048x49xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_189[%arg1, %arg2, %arg3] : memref<1x512x49xf32>
        }
      }
    }
    %alloc_190 = memref.alloc() {alignment = 16 : i64} : memref<1x512x49xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 49 {
          %124 = affine.load %alloc_189[%c0, %arg2, %arg3] : memref<1x512x49xf32>
          %125 = affine.load %1[%c0, %arg2, %c0] : memref<1x512x1xf32>
          %126 = arith.addf %124, %125 : f32
          affine.store %126, %alloc_190[%arg1, %arg2, %arg3] : memref<1x512x49xf32>
        }
      }
    }
    %reinterpret_cast_191 = memref.reinterpret_cast %alloc_190 to offset: [0], sizes: [1, 512, 7, 7], strides: [25088, 49, 7, 1] : memref<1x512x49xf32> to memref<1x512x7x7xf32>
    %alloc_192 = memref.alloc() {alignment = 16 : i64} : memref<1x512x7x7xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 7 {
          affine.for %arg4 = 0 to 7 {
            %124 = affine.load %reinterpret_cast_191[%arg1, %arg2, %arg3, %arg4] : memref<1x512x7x7xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_192[%arg1, %arg2, %arg3, %arg4] : memref<1x512x7x7xf32>
          }
        }
      }
    }
    %alloc_193 = memref.alloc() {alignment = 16 : i64} : memref<1x512x7x7xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 512 {
          %124 = affine.apply #map22(%arg2, %arg3)
          affine.for %arg4 = 0 to 7 {
            affine.for %arg5 = 0 to 7 {
              %125 = affine.for %arg6 = 0 to 512 iter_args(%arg7 = %cst_1) -> (f32) {
                %128 = affine.for %arg8 = max #map9(%arg4) to min #map46(%arg4) iter_args(%arg9 = %arg7) -> (f32) {
                  %129 = affine.for %arg10 = max #map11(%arg4, %arg5) to min #map47(%arg4, %arg5) iter_args(%arg11 = %arg9) -> (f32) {
                    %130 = affine.apply #map37(%arg6, %arg2)
                    %131 = affine.apply #map14(%arg8, %arg4)
                    %132 = affine.apply #map14(%arg10, %arg5)
                    %133 = affine.load %alloc_192[%arg1, %130, %131, %132] : memref<1x512x7x7xf32>
                    %134 = affine.load %52[%124, %arg6, %arg8, %arg10] : memref<512x512x3x3xf32>
                    %135 = arith.mulf %133, %134 : f32
                    %136 = arith.addf %arg11, %135 : f32
                    affine.yield %136 : f32
                  }
                  affine.yield %129 : f32
                }
                affine.yield %128 : f32
              }
              %126 = affine.load %51[%124] : memref<512xf32>
              %127 = arith.addf %125, %126 : f32
              affine.store %127, %alloc_193[%arg1, %124, %arg4, %arg5] : memref<1x512x7x7xf32>
            }
          }
        }
      }
    }
    %alloc_194 = memref.alloc() {alignment = 16 : i64} : memref<1x512x7x7xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 512 {
        affine.for %arg3 = 0 to 7 {
          affine.for %arg4 = 0 to 7 {
            %124 = affine.load %alloc_193[%arg1, %arg2, %arg3, %arg4] : memref<1x512x7x7xf32>
            %125 = arith.maxf %124, %cst_1 : f32
            affine.store %125, %alloc_194[%arg1, %arg2, %arg3, %arg4] : memref<1x512x7x7xf32>
          }
        }
      }
    }
    %reinterpret_cast_195 = memref.reinterpret_cast %alloc_194 to offset: [0], sizes: [1, 512, 49], strides: [25088, 49, 1] : memref<1x512x7x7xf32> to memref<1x512x49xf32>
    %alloc_196 = memref.alloc() {alignment = 16 : i64} : memref<1x2048x49xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 2048 {
        affine.for %arg3 = 0 to 49 {
          %124 = affine.for %arg4 = 0 to 512 iter_args(%arg5 = %cst_1) -> (f32) {
            %125 = affine.load %0[%arg2, %arg4] : memref<2048x512xf32>
            %126 = affine.load %reinterpret_cast_195[%arg1, %arg4, %arg3] : memref<1x512x49xf32>
            %127 = arith.mulf %125, %126 : f32
            %128 = arith.addf %arg5, %127 : f32
            affine.yield %128 : f32
          }
          affine.store %124, %alloc_196[%arg1, %arg2, %arg3] : memref<1x2048x49xf32>
        }
      }
    }
    %reinterpret_cast_197 = memref.reinterpret_cast %alloc_196 to offset: [0], sizes: [1, 2048, 7, 7], strides: [100352, 49, 7, 1] : memref<1x2048x49xf32> to memref<1x2048x7x7xf32>
    %alloc_198 = memref.alloc() {alignment = 16 : i64} : memref<1x2048x7x7xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 2048 {
        affine.for %arg3 = 0 to 7 {
          affine.for %arg4 = 0 to 7 {
            %124 = affine.load %reinterpret_cast_197[%c0, %arg2, %arg3, %arg4] : memref<1x2048x7x7xf32>
            %125 = affine.load %alloc_186[%c0, %arg2, %arg3, %arg4] : memref<1x2048x7x7xf32>
            %126 = arith.addf %124, %125 : f32
            %127 = affine.load %50[%arg2, %c0, %c0] : memref<2048x1x1xf32>
            %128 = arith.mulf %126, %127 : f32
            %129 = affine.load %49[%arg2, %c0, %c0] : memref<2048x1x1xf32>
            %130 = arith.addf %128, %129 : f32
            %131 = arith.maxf %130, %cst_1 : f32
            affine.store %131, %alloc_198[%arg1, %arg2, %arg3, %arg4] : memref<1x2048x7x7xf32>
          }
        }
      }
    }
    %alloc_199 = memref.alloc() {alignment = 16 : i64} : memref<1x2048x1x1xf32>
    %c1_200 = arith.constant 1 : index
    %c2048 = arith.constant 2048 : index
    %c1_201 = arith.constant 1 : index
    %c1_202 = arith.constant 1 : index
    %c0_203 = arith.constant 0 : index
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 2048 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 1 {
            affine.store %cst_1, %alloc_199[%arg1, %arg2, %arg3, %arg4] : memref<1x2048x1x1xf32>
          }
        }
      }
    }
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 2048 {
        affine.for %arg3 = 0 to 7 {
          affine.for %arg4 = 0 to 7 {
            %124 = affine.load %alloc_198[%arg1, %arg2, %arg3, %arg4] : memref<1x2048x7x7xf32>
            %125 = affine.load %alloc_199[%arg1, %arg2, %c0, %c0] : memref<1x2048x1x1xf32>
            %126 = arith.addf %125, %124 : f32
            affine.store %126, %alloc_199[%arg1, %arg2, %c0, %c0] : memref<1x2048x1x1xf32>
          }
        }
      }
    }
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 2048 {
        affine.for %arg3 = 0 to 1 {
          affine.for %arg4 = 0 to 1 {
            %124 = affine.load %alloc_199[%arg1, %arg2, %arg3, %arg4] : memref<1x2048x1x1xf32>
            %125 = arith.divf %124, %cst : f32
            affine.store %125, %alloc_199[%arg1, %arg2, %arg3, %arg4] : memref<1x2048x1x1xf32>
          }
        }
      }
    }
    %reinterpret_cast_204 = memref.reinterpret_cast %alloc_199 to offset: [0], sizes: [1, 2048], strides: [2048, 1] : memref<1x2048x1x1xf32> to memref<1x2048xf32>
    %alloc_205 = memref.alloc() {alignment = 128 : i64} : memref<1x1000xf32>
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1000 {
        %alloca_206 = memref.alloca() : memref<f32>
        affine.store %cst_1, %alloca_206[] : memref<f32>
        affine.for %arg3 = 0 to 2048 {
          %127 = affine.load %reinterpret_cast_204[%arg1, %arg3] : memref<1x2048xf32>
          %128 = affine.load %121[%arg2, %arg3] : memref<1000x2048xf32>
          %129 = arith.mulf %127, %128 : f32
          %130 = affine.load %alloca_206[] : memref<f32>
          %131 = arith.addf %129, %130 : f32
          affine.store %131, %alloca_206[] : memref<f32>
        }
        %124 = affine.load %alloca_206[] : memref<f32>
        %125 = affine.load %122[%arg2] : memref<1000xf32>
        %126 = arith.addf %124, %125 : f32
        affine.store %126, %alloc_205[%arg1, %arg2] : memref<1x1000xf32>
      }
    }
    return %alloc_205 : memref<1x1000xf32>
  }

}

