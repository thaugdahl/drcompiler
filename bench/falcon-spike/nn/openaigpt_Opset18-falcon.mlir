#map = affine_map<(d0) -> (d0 + 768)>
#map1 = affine_map<(d0) -> (d0 + 1536)>
#map2 = affine_map<(d0, d1, d2) -> (d0 * 98304 + d1 * 768 + d2 * 64)>
#map3 = affine_map<(d0, d1, d2) -> (d0 * 98304 + d1 * 64 + d2 * 8192)>
#map4 = affine_map<(d0) -> (d0 + 1)>
#map5 = affine_map<(d0) -> (d0 + 2)>
#map6 = affine_map<(d0) -> (d0 + 3)>
#map7 = affine_map<(d0) -> (d0 + 4)>
#map8 = affine_map<(d0) -> (d0 + 5)>
#map9 = affine_map<(d0) -> (d0 + 6)>
#map10 = affine_map<(d0) -> (d0 + 7)>
#map11 = affine_map<(d0, d1, d2) -> (d0 * 98304 + d1 * 8192 + d2 * 64)>
#map12 = affine_map<(d0, d1, d2) -> (d0 * 98304 + d1 * 64 + d2 * 768)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "openaigpt_opset18"} {
  memref.global "private" constant @constant_1 : memref<1x128x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_7 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_8 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_9 : memref<f32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_21 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_22 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_23 : memref<f32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_35 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_36 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_37 : memref<f32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_49 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_50 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_51 : memref<f32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_63 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_64 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_65 : memref<f32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_77 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_78 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_79 : memref<f32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_91 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_92 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_93 : memref<f32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_105 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_106 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_107 : memref<f32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_119 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_120 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_121 : memref<f32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_133 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_134 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_135 : memref<f32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_147 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_148 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_149 : memref<f32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_161 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_162 : memref<1x1x128x128xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_163 : memref<f32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_170 : memref<f32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_171 : memref<f32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_174 : memref<40478x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_175 : memref<768x2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_176 : memref<2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_177 : memref<768x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_178 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_179 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_180 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_181 : memref<768x3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_182 : memref<3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_183 : memref<3072x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_184 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_185 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_186 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_187 : memref<768x2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_188 : memref<2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_189 : memref<768x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_190 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_191 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_192 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_193 : memref<768x3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_194 : memref<3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_195 : memref<3072x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_196 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_197 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_198 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_199 : memref<768x2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_200 : memref<2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_201 : memref<768x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_202 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_203 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_204 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_205 : memref<768x3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_206 : memref<3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_207 : memref<3072x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_208 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_209 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_210 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_211 : memref<768x2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_212 : memref<2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_213 : memref<768x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_214 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_215 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_216 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_217 : memref<768x3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_218 : memref<3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_219 : memref<3072x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_220 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_221 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_222 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_223 : memref<768x2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_224 : memref<2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_225 : memref<768x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_226 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_227 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_228 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_229 : memref<768x3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_230 : memref<3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_231 : memref<3072x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_232 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_233 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_234 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_235 : memref<768x2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_236 : memref<2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_237 : memref<768x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_238 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_239 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_240 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_241 : memref<768x3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_242 : memref<3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_243 : memref<3072x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_244 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_245 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_246 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_247 : memref<768x2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_248 : memref<2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_249 : memref<768x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_250 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_251 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_252 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_253 : memref<768x3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_254 : memref<3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_255 : memref<3072x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_256 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_257 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_258 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_259 : memref<768x2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_260 : memref<2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_261 : memref<768x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_262 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_263 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_264 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_265 : memref<768x3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_266 : memref<3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_267 : memref<3072x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_268 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_269 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_270 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_271 : memref<768x2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_272 : memref<2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_273 : memref<768x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_274 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_275 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_276 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_277 : memref<768x3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_278 : memref<3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_279 : memref<3072x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_280 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_281 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_282 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_283 : memref<768x2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_284 : memref<2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_285 : memref<768x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_286 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_287 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_288 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_289 : memref<768x3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_290 : memref<3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_291 : memref<3072x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_292 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_293 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_294 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_295 : memref<768x2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_296 : memref<2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_297 : memref<768x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_298 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_299 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_300 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_301 : memref<768x3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_302 : memref<3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_303 : memref<3072x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_304 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_305 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_306 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_307 : memref<768x2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_308 : memref<2304xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_309 : memref<768x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_310 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_311 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_312 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_313 : memref<768x3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_314 : memref<3072xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_315 : memref<3072x768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_316 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_317 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_318 : memref<768xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_319 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_321 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_323 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_325 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_327 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_329 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_331 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_333 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_335 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_337 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_339 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_341 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_343 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_345 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_347 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_349 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_351 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_353 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_355 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_357 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_359 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_361 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_363 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  memref.global "private" constant @constant_365 : memref<1xf32> = dense_resource<__elided__> {alignment = 64 : i64}
  func.func @main_graph(%arg0: memref<1x128xi64> {onnx.name = "input_ids"}, %arg1: memref<1x128xf32> {onnx.name = "attention_mask"}) -> (memref<1x128x768xf32> {onnx.name = "2065"}) attributes {llvm.emit_c_interface} {
    %cst = arith.constant 7.680000e+02 : f32
    %c64_i64 = arith.constant 64 : i64
    %cst_0 = arith.constant 0.797884583 : f32
    %cst_1 = arith.constant 4.471500e-02 : f32
    %cst_2 = arith.constant 3.000000e+00 : f32
    %cst_3 = arith.constant 5.000000e-01 : f32
    %cst_4 = arith.constant 0xFF800000 : f32
    %cst_5 = arith.constant 0.000000e+00 : f32
    %cst_6 = arith.constant 1.000000e+00 : f32
    %c40478 = arith.constant 40478 : index
    %c0 = arith.constant 0 : index
    %0 = memref.get_global @constant_1 : memref<1x128x768xf32>
    %1 = memref.get_global @constant_7 : memref<1x1x128x128xf32>
    %2 = memref.get_global @constant_8 : memref<1x1x128x128xf32>
    %3 = memref.get_global @constant_9 : memref<f32>
    %4 = memref.get_global @constant_21 : memref<1x1x128x128xf32>
    %5 = memref.get_global @constant_22 : memref<1x1x128x128xf32>
    %6 = memref.get_global @constant_23 : memref<f32>
    %7 = memref.get_global @constant_35 : memref<1x1x128x128xf32>
    %8 = memref.get_global @constant_36 : memref<1x1x128x128xf32>
    %9 = memref.get_global @constant_37 : memref<f32>
    %10 = memref.get_global @constant_49 : memref<1x1x128x128xf32>
    %11 = memref.get_global @constant_50 : memref<1x1x128x128xf32>
    %12 = memref.get_global @constant_51 : memref<f32>
    %13 = memref.get_global @constant_63 : memref<1x1x128x128xf32>
    %14 = memref.get_global @constant_64 : memref<1x1x128x128xf32>
    %15 = memref.get_global @constant_65 : memref<f32>
    %16 = memref.get_global @constant_77 : memref<1x1x128x128xf32>
    %17 = memref.get_global @constant_78 : memref<1x1x128x128xf32>
    %18 = memref.get_global @constant_79 : memref<f32>
    %19 = memref.get_global @constant_91 : memref<1x1x128x128xf32>
    %20 = memref.get_global @constant_92 : memref<1x1x128x128xf32>
    %21 = memref.get_global @constant_93 : memref<f32>
    %22 = memref.get_global @constant_105 : memref<1x1x128x128xf32>
    %23 = memref.get_global @constant_106 : memref<1x1x128x128xf32>
    %24 = memref.get_global @constant_107 : memref<f32>
    %25 = memref.get_global @constant_119 : memref<1x1x128x128xf32>
    %26 = memref.get_global @constant_120 : memref<1x1x128x128xf32>
    %27 = memref.get_global @constant_121 : memref<f32>
    %28 = memref.get_global @constant_133 : memref<1x1x128x128xf32>
    %29 = memref.get_global @constant_134 : memref<1x1x128x128xf32>
    %30 = memref.get_global @constant_135 : memref<f32>
    %31 = memref.get_global @constant_147 : memref<1x1x128x128xf32>
    %32 = memref.get_global @constant_148 : memref<1x1x128x128xf32>
    %33 = memref.get_global @constant_149 : memref<f32>
    %34 = memref.get_global @constant_161 : memref<1x1x128x128xf32>
    %35 = memref.get_global @constant_162 : memref<1x1x128x128xf32>
    %36 = memref.get_global @constant_163 : memref<f32>
    %37 = memref.get_global @constant_170 : memref<f32>
    %38 = memref.get_global @constant_171 : memref<f32>
    %39 = memref.get_global @constant_174 : memref<40478x768xf32>
    %40 = memref.get_global @constant_175 : memref<768x2304xf32>
    %41 = memref.get_global @constant_176 : memref<2304xf32>
    %42 = memref.get_global @constant_177 : memref<768x768xf32>
    %43 = memref.get_global @constant_178 : memref<768xf32>
    %44 = memref.get_global @constant_179 : memref<768xf32>
    %45 = memref.get_global @constant_180 : memref<768xf32>
    %46 = memref.get_global @constant_181 : memref<768x3072xf32>
    %47 = memref.get_global @constant_182 : memref<3072xf32>
    %48 = memref.get_global @constant_183 : memref<3072x768xf32>
    %49 = memref.get_global @constant_184 : memref<768xf32>
    %50 = memref.get_global @constant_185 : memref<768xf32>
    %51 = memref.get_global @constant_186 : memref<768xf32>
    %52 = memref.get_global @constant_187 : memref<768x2304xf32>
    %53 = memref.get_global @constant_188 : memref<2304xf32>
    %54 = memref.get_global @constant_189 : memref<768x768xf32>
    %55 = memref.get_global @constant_190 : memref<768xf32>
    %56 = memref.get_global @constant_191 : memref<768xf32>
    %57 = memref.get_global @constant_192 : memref<768xf32>
    %58 = memref.get_global @constant_193 : memref<768x3072xf32>
    %59 = memref.get_global @constant_194 : memref<3072xf32>
    %60 = memref.get_global @constant_195 : memref<3072x768xf32>
    %61 = memref.get_global @constant_196 : memref<768xf32>
    %62 = memref.get_global @constant_197 : memref<768xf32>
    %63 = memref.get_global @constant_198 : memref<768xf32>
    %64 = memref.get_global @constant_199 : memref<768x2304xf32>
    %65 = memref.get_global @constant_200 : memref<2304xf32>
    %66 = memref.get_global @constant_201 : memref<768x768xf32>
    %67 = memref.get_global @constant_202 : memref<768xf32>
    %68 = memref.get_global @constant_203 : memref<768xf32>
    %69 = memref.get_global @constant_204 : memref<768xf32>
    %70 = memref.get_global @constant_205 : memref<768x3072xf32>
    %71 = memref.get_global @constant_206 : memref<3072xf32>
    %72 = memref.get_global @constant_207 : memref<3072x768xf32>
    %73 = memref.get_global @constant_208 : memref<768xf32>
    %74 = memref.get_global @constant_209 : memref<768xf32>
    %75 = memref.get_global @constant_210 : memref<768xf32>
    %76 = memref.get_global @constant_211 : memref<768x2304xf32>
    %77 = memref.get_global @constant_212 : memref<2304xf32>
    %78 = memref.get_global @constant_213 : memref<768x768xf32>
    %79 = memref.get_global @constant_214 : memref<768xf32>
    %80 = memref.get_global @constant_215 : memref<768xf32>
    %81 = memref.get_global @constant_216 : memref<768xf32>
    %82 = memref.get_global @constant_217 : memref<768x3072xf32>
    %83 = memref.get_global @constant_218 : memref<3072xf32>
    %84 = memref.get_global @constant_219 : memref<3072x768xf32>
    %85 = memref.get_global @constant_220 : memref<768xf32>
    %86 = memref.get_global @constant_221 : memref<768xf32>
    %87 = memref.get_global @constant_222 : memref<768xf32>
    %88 = memref.get_global @constant_223 : memref<768x2304xf32>
    %89 = memref.get_global @constant_224 : memref<2304xf32>
    %90 = memref.get_global @constant_225 : memref<768x768xf32>
    %91 = memref.get_global @constant_226 : memref<768xf32>
    %92 = memref.get_global @constant_227 : memref<768xf32>
    %93 = memref.get_global @constant_228 : memref<768xf32>
    %94 = memref.get_global @constant_229 : memref<768x3072xf32>
    %95 = memref.get_global @constant_230 : memref<3072xf32>
    %96 = memref.get_global @constant_231 : memref<3072x768xf32>
    %97 = memref.get_global @constant_232 : memref<768xf32>
    %98 = memref.get_global @constant_233 : memref<768xf32>
    %99 = memref.get_global @constant_234 : memref<768xf32>
    %100 = memref.get_global @constant_235 : memref<768x2304xf32>
    %101 = memref.get_global @constant_236 : memref<2304xf32>
    %102 = memref.get_global @constant_237 : memref<768x768xf32>
    %103 = memref.get_global @constant_238 : memref<768xf32>
    %104 = memref.get_global @constant_239 : memref<768xf32>
    %105 = memref.get_global @constant_240 : memref<768xf32>
    %106 = memref.get_global @constant_241 : memref<768x3072xf32>
    %107 = memref.get_global @constant_242 : memref<3072xf32>
    %108 = memref.get_global @constant_243 : memref<3072x768xf32>
    %109 = memref.get_global @constant_244 : memref<768xf32>
    %110 = memref.get_global @constant_245 : memref<768xf32>
    %111 = memref.get_global @constant_246 : memref<768xf32>
    %112 = memref.get_global @constant_247 : memref<768x2304xf32>
    %113 = memref.get_global @constant_248 : memref<2304xf32>
    %114 = memref.get_global @constant_249 : memref<768x768xf32>
    %115 = memref.get_global @constant_250 : memref<768xf32>
    %116 = memref.get_global @constant_251 : memref<768xf32>
    %117 = memref.get_global @constant_252 : memref<768xf32>
    %118 = memref.get_global @constant_253 : memref<768x3072xf32>
    %119 = memref.get_global @constant_254 : memref<3072xf32>
    %120 = memref.get_global @constant_255 : memref<3072x768xf32>
    %121 = memref.get_global @constant_256 : memref<768xf32>
    %122 = memref.get_global @constant_257 : memref<768xf32>
    %123 = memref.get_global @constant_258 : memref<768xf32>
    %124 = memref.get_global @constant_259 : memref<768x2304xf32>
    %125 = memref.get_global @constant_260 : memref<2304xf32>
    %126 = memref.get_global @constant_261 : memref<768x768xf32>
    %127 = memref.get_global @constant_262 : memref<768xf32>
    %128 = memref.get_global @constant_263 : memref<768xf32>
    %129 = memref.get_global @constant_264 : memref<768xf32>
    %130 = memref.get_global @constant_265 : memref<768x3072xf32>
    %131 = memref.get_global @constant_266 : memref<3072xf32>
    %132 = memref.get_global @constant_267 : memref<3072x768xf32>
    %133 = memref.get_global @constant_268 : memref<768xf32>
    %134 = memref.get_global @constant_269 : memref<768xf32>
    %135 = memref.get_global @constant_270 : memref<768xf32>
    %136 = memref.get_global @constant_271 : memref<768x2304xf32>
    %137 = memref.get_global @constant_272 : memref<2304xf32>
    %138 = memref.get_global @constant_273 : memref<768x768xf32>
    %139 = memref.get_global @constant_274 : memref<768xf32>
    %140 = memref.get_global @constant_275 : memref<768xf32>
    %141 = memref.get_global @constant_276 : memref<768xf32>
    %142 = memref.get_global @constant_277 : memref<768x3072xf32>
    %143 = memref.get_global @constant_278 : memref<3072xf32>
    %144 = memref.get_global @constant_279 : memref<3072x768xf32>
    %145 = memref.get_global @constant_280 : memref<768xf32>
    %146 = memref.get_global @constant_281 : memref<768xf32>
    %147 = memref.get_global @constant_282 : memref<768xf32>
    %148 = memref.get_global @constant_283 : memref<768x2304xf32>
    %149 = memref.get_global @constant_284 : memref<2304xf32>
    %150 = memref.get_global @constant_285 : memref<768x768xf32>
    %151 = memref.get_global @constant_286 : memref<768xf32>
    %152 = memref.get_global @constant_287 : memref<768xf32>
    %153 = memref.get_global @constant_288 : memref<768xf32>
    %154 = memref.get_global @constant_289 : memref<768x3072xf32>
    %155 = memref.get_global @constant_290 : memref<3072xf32>
    %156 = memref.get_global @constant_291 : memref<3072x768xf32>
    %157 = memref.get_global @constant_292 : memref<768xf32>
    %158 = memref.get_global @constant_293 : memref<768xf32>
    %159 = memref.get_global @constant_294 : memref<768xf32>
    %160 = memref.get_global @constant_295 : memref<768x2304xf32>
    %161 = memref.get_global @constant_296 : memref<2304xf32>
    %162 = memref.get_global @constant_297 : memref<768x768xf32>
    %163 = memref.get_global @constant_298 : memref<768xf32>
    %164 = memref.get_global @constant_299 : memref<768xf32>
    %165 = memref.get_global @constant_300 : memref<768xf32>
    %166 = memref.get_global @constant_301 : memref<768x3072xf32>
    %167 = memref.get_global @constant_302 : memref<3072xf32>
    %168 = memref.get_global @constant_303 : memref<3072x768xf32>
    %169 = memref.get_global @constant_304 : memref<768xf32>
    %170 = memref.get_global @constant_305 : memref<768xf32>
    %171 = memref.get_global @constant_306 : memref<768xf32>
    %172 = memref.get_global @constant_307 : memref<768x2304xf32>
    %173 = memref.get_global @constant_308 : memref<2304xf32>
    %174 = memref.get_global @constant_309 : memref<768x768xf32>
    %175 = memref.get_global @constant_310 : memref<768xf32>
    %176 = memref.get_global @constant_311 : memref<768xf32>
    %177 = memref.get_global @constant_312 : memref<768xf32>
    %178 = memref.get_global @constant_313 : memref<768x3072xf32>
    %179 = memref.get_global @constant_314 : memref<3072xf32>
    %180 = memref.get_global @constant_315 : memref<3072x768xf32>
    %181 = memref.get_global @constant_316 : memref<768xf32>
    %182 = memref.get_global @constant_317 : memref<768xf32>
    %183 = memref.get_global @constant_318 : memref<768xf32>
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [1, 1, 1, 128], strides: [128, 128, 128, 1] : memref<1x128xf32> to memref<1x1x1x128xf32>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x1x1x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 1 {
        affine.for %arg4 = 0 to 1 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %38[] : memref<f32>
            %209 = affine.load %reinterpret_cast[%c0, %c0, %c0, %arg5] : memref<1x1x1x128xf32>
            %210 = arith.subf %208, %209 : f32
            %211 = affine.load %37[] : memref<f32>
            %212 = arith.mulf %210, %211 : f32
            affine.store %212, %alloc[%arg2, %arg3, %arg4, %arg5] : memref<1x1x1x128xf32>
          }
        }
      }
    }
    %alloc_7 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %arg0[%arg2, %arg3] : memref<1x128xi64>
          %209 = arith.index_cast %208 : i64 to index
          %210 = arith.addi %209, %c40478 : index
          %211 = arith.cmpi slt, %209, %c0 : index
          %212 = arith.select %211, %210, %209 : index
          %213 = memref.load %39[%212, %arg4] : memref<40478x768xf32>
          affine.store %213, %alloc_7[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_8 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_7[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %0[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_8[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_9 = memref.reinterpret_cast %alloc_8 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_10 = memref.alloc() {alignment = 128 : i64} : memref<128x2304xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 2304 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_9[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %40[%arg4, %arg3] : memref<768x2304xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %41[%arg3] : memref<2304xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_10[%arg2, %arg3] : memref<128x2304xf32>
      }
    }
    %reinterpret_cast_11 = memref.reinterpret_cast %alloc_10 to offset: [0], sizes: [1, 128, 2304], strides: [294912, 2304, 1] : memref<128x2304xf32> to memref<1x128x2304xf32>
    %alloc_12 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_13 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_14 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %reinterpret_cast_11[%arg2, %arg3, %arg4] : memref<1x128x2304xf32>
          affine.store %208, %alloc_12[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map(%arg4)
          %209 = affine.load %reinterpret_cast_11[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_13[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map1(%arg4)
          %209 = affine.load %reinterpret_cast_11[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_14[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_15 = memref.reinterpret_cast %alloc_12 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_16 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_16, %reinterpret_cast_15, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_17 = memref.reinterpret_cast %alloc_13 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_18 = memref.alloc() {alignment = 16 : i64} : memref<1x12x64x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 64 {
          affine.for %arg5 = 0 to 128 step 8 {
            %208 = affine.load %reinterpret_cast_17[%arg2, %arg5, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %208, %alloc_18[%arg2, %arg3, %arg4, %arg5] : memref<1x12x64x128xf32>
            %209 = affine.apply #map4(%arg5)
            %210 = affine.load %reinterpret_cast_17[%arg2, %209, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %210, %alloc_18[%arg2, %arg3, %arg4, %209] : memref<1x12x64x128xf32>
            %211 = affine.apply #map5(%arg5)
            %212 = affine.load %reinterpret_cast_17[%arg2, %211, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %212, %alloc_18[%arg2, %arg3, %arg4, %211] : memref<1x12x64x128xf32>
            %213 = affine.apply #map6(%arg5)
            %214 = affine.load %reinterpret_cast_17[%arg2, %213, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %214, %alloc_18[%arg2, %arg3, %arg4, %213] : memref<1x12x64x128xf32>
            %215 = affine.apply #map7(%arg5)
            %216 = affine.load %reinterpret_cast_17[%arg2, %215, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %216, %alloc_18[%arg2, %arg3, %arg4, %215] : memref<1x12x64x128xf32>
            %217 = affine.apply #map8(%arg5)
            %218 = affine.load %reinterpret_cast_17[%arg2, %217, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %218, %alloc_18[%arg2, %arg3, %arg4, %217] : memref<1x12x64x128xf32>
            %219 = affine.apply #map9(%arg5)
            %220 = affine.load %reinterpret_cast_17[%arg2, %219, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %220, %alloc_18[%arg2, %arg3, %arg4, %219] : memref<1x12x64x128xf32>
            %221 = affine.apply #map10(%arg5)
            %222 = affine.load %reinterpret_cast_17[%arg2, %221, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %222, %alloc_18[%arg2, %arg3, %arg4, %221] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %reinterpret_cast_19 = memref.reinterpret_cast %alloc_14 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_20 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_20, %reinterpret_cast_19, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %alloc_21 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_16[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x64xf32>
              %210 = affine.load %alloc_18[%arg2, %arg3, %arg6, %arg5] : memref<1x12x64x128xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_21[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_22 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_21[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %36[] : memref<f32>
            %210 = arith.divf %208, %209 : f32
            %211 = affine.load %35[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.mulf %210, %211 : f32
            affine.store %212, %alloc_22[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_23 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_22[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %alloc[%c0, %c0, %c0, %arg5] : memref<1x1x1x128xf32>
            %210 = arith.addf %208, %209 : f32
            %211 = affine.load %34[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.addf %210, %211 : f32
            affine.store %212, %alloc_23[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_24 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_4) -> (f32) {
            %210 = affine.load %alloc_23[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.maxf %arg6, %210 : f32
            affine.yield %211 : f32
          }
          %209 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_5) -> (f32) {
            %210 = affine.load %alloc_23[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.subf %210, %208 : f32
            %212 = math.exp %211 : f32
            %213 = arith.addf %arg6, %212 : f32
            affine.store %212, %alloc_24[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            affine.yield %213 : f32
          }
          affine.for %arg5 = 0 to 128 {
            %210 = affine.load %alloc_24[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.divf %210, %209 : f32
            affine.store %211, %alloc_24[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_25 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 64 {
            %208 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_24[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x128xf32>
              %210 = affine.load %alloc_20[%arg2, %arg3, %arg6, %arg5] : memref<1x12x128x64xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_25[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_26 = memref.alloc() {alignment = 16 : i64} : memref<1x128x12x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.apply #map11(%arg2, %arg3, %arg4)
          %209 = affine.apply #map12(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_26, %alloc_25, %c64_i64, %209, %208) : (memref<1x128x12x64xf32>, memref<1x12x128x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_27 = memref.reinterpret_cast %alloc_26 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x12x64xf32> to memref<128x768xf32>
    %alloc_28 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_27[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %42[%arg4, %arg3] : memref<768x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %43[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_28[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_29 = memref.reinterpret_cast %alloc_28 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_30 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_8[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_29[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_30[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %184 = memref.get_global @constant_319 : memref<1xf32>
    %alloc_31 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %c1_32 = arith.constant 1 : index
    %c0_33 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_31[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_30[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_31[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_31[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_31[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_31[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_34 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_31[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_31[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_34[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_35 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_30[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_30[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_35[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_36 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_37 = arith.constant 1 : index
    %c128_38 = arith.constant 128 : index
    %c1_39 = arith.constant 1 : index
    %c0_40 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_36[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_35[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_36[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_36[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_36[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_36[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_41 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_36[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_34[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %184[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_41[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_42 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_30[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_31[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_42[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_43 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_42[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_41[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %44[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %45[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_43[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_44 = memref.reinterpret_cast %alloc_43 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_45 = memref.alloc() {alignment = 128 : i64} : memref<128x3072xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 3072 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_44[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %46[%arg4, %arg3] : memref<768x3072xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %47[%arg3] : memref<3072xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_45[%arg2, %arg3] : memref<128x3072xf32>
      }
    }
    %reinterpret_cast_46 = memref.reinterpret_cast %alloc_45 to offset: [0], sizes: [1, 128, 3072], strides: [393216, 3072, 1] : memref<128x3072xf32> to memref<1x128x3072xf32>
    %alloc_47 = memref.alloc() {alignment = 16 : i64} : memref<1x128x3072xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 3072 {
          %208 = affine.load %reinterpret_cast_46[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
          %209 = arith.mulf %208, %cst_3 : f32
          %210 = math.powf %208, %cst_2 : f32
          %211 = arith.mulf %210, %cst_1 : f32
          %212 = arith.addf %208, %211 : f32
          %213 = arith.mulf %212, %cst_0 : f32
          %214 = math.tanh %213 : f32
          %215 = arith.addf %214, %cst_6 : f32
          %216 = arith.mulf %209, %215 : f32
          affine.store %216, %alloc_47[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
        }
      }
    }
    %reinterpret_cast_48 = memref.reinterpret_cast %alloc_47 to offset: [0], sizes: [128, 3072], strides: [3072, 1] : memref<1x128x3072xf32> to memref<128x3072xf32>
    %alloc_49 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 3072 {
          %211 = affine.load %reinterpret_cast_48[%arg2, %arg4] : memref<128x3072xf32>
          %212 = affine.load %48[%arg4, %arg3] : memref<3072x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %49[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_49[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_50 = memref.reinterpret_cast %alloc_49 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_51 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_43[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_50[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_51[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %185 = memref.get_global @constant_321 : memref<1xf32>
    %alloc_52 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_53 = arith.constant 1 : index
    %c128_54 = arith.constant 128 : index
    %c1_55 = arith.constant 1 : index
    %c0_56 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_52[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_51[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_52[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_52[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_52[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_52[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_57 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_52[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_52[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_57[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_58 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_51[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_51[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_58[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_59 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_60 = arith.constant 1 : index
    %c128_61 = arith.constant 128 : index
    %c1_62 = arith.constant 1 : index
    %c0_63 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_59[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_58[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_59[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_59[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_59[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_59[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_64 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_59[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_57[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %185[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_64[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_65 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_51[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_52[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_65[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_66 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_65[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_64[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %50[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %51[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_66[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_67 = memref.reinterpret_cast %alloc_66 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_68 = memref.alloc() {alignment = 128 : i64} : memref<128x2304xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 2304 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_67[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %52[%arg4, %arg3] : memref<768x2304xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %53[%arg3] : memref<2304xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_68[%arg2, %arg3] : memref<128x2304xf32>
      }
    }
    %reinterpret_cast_69 = memref.reinterpret_cast %alloc_68 to offset: [0], sizes: [1, 128, 2304], strides: [294912, 2304, 1] : memref<128x2304xf32> to memref<1x128x2304xf32>
    %alloc_70 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_71 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_72 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %reinterpret_cast_69[%arg2, %arg3, %arg4] : memref<1x128x2304xf32>
          affine.store %208, %alloc_70[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map(%arg4)
          %209 = affine.load %reinterpret_cast_69[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_71[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map1(%arg4)
          %209 = affine.load %reinterpret_cast_69[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_72[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_73 = memref.reinterpret_cast %alloc_70 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_74 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_74, %reinterpret_cast_73, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_75 = memref.reinterpret_cast %alloc_71 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_76 = memref.alloc() {alignment = 16 : i64} : memref<1x12x64x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 64 {
          affine.for %arg5 = 0 to 128 step 8 {
            %208 = affine.load %reinterpret_cast_75[%arg2, %arg5, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %208, %alloc_76[%arg2, %arg3, %arg4, %arg5] : memref<1x12x64x128xf32>
            %209 = affine.apply #map4(%arg5)
            %210 = affine.load %reinterpret_cast_75[%arg2, %209, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %210, %alloc_76[%arg2, %arg3, %arg4, %209] : memref<1x12x64x128xf32>
            %211 = affine.apply #map5(%arg5)
            %212 = affine.load %reinterpret_cast_75[%arg2, %211, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %212, %alloc_76[%arg2, %arg3, %arg4, %211] : memref<1x12x64x128xf32>
            %213 = affine.apply #map6(%arg5)
            %214 = affine.load %reinterpret_cast_75[%arg2, %213, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %214, %alloc_76[%arg2, %arg3, %arg4, %213] : memref<1x12x64x128xf32>
            %215 = affine.apply #map7(%arg5)
            %216 = affine.load %reinterpret_cast_75[%arg2, %215, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %216, %alloc_76[%arg2, %arg3, %arg4, %215] : memref<1x12x64x128xf32>
            %217 = affine.apply #map8(%arg5)
            %218 = affine.load %reinterpret_cast_75[%arg2, %217, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %218, %alloc_76[%arg2, %arg3, %arg4, %217] : memref<1x12x64x128xf32>
            %219 = affine.apply #map9(%arg5)
            %220 = affine.load %reinterpret_cast_75[%arg2, %219, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %220, %alloc_76[%arg2, %arg3, %arg4, %219] : memref<1x12x64x128xf32>
            %221 = affine.apply #map10(%arg5)
            %222 = affine.load %reinterpret_cast_75[%arg2, %221, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %222, %alloc_76[%arg2, %arg3, %arg4, %221] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %reinterpret_cast_77 = memref.reinterpret_cast %alloc_72 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_78 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_78, %reinterpret_cast_77, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %alloc_79 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_74[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x64xf32>
              %210 = affine.load %alloc_76[%arg2, %arg3, %arg6, %arg5] : memref<1x12x64x128xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_79[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_80 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_79[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %33[] : memref<f32>
            %210 = arith.divf %208, %209 : f32
            %211 = affine.load %32[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.mulf %210, %211 : f32
            affine.store %212, %alloc_80[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_81 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_80[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %alloc[%c0, %c0, %c0, %arg5] : memref<1x1x1x128xf32>
            %210 = arith.addf %208, %209 : f32
            %211 = affine.load %31[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.addf %210, %211 : f32
            affine.store %212, %alloc_81[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_82 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_4) -> (f32) {
            %210 = affine.load %alloc_81[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.maxf %arg6, %210 : f32
            affine.yield %211 : f32
          }
          %209 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_5) -> (f32) {
            %210 = affine.load %alloc_81[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.subf %210, %208 : f32
            %212 = math.exp %211 : f32
            %213 = arith.addf %arg6, %212 : f32
            affine.store %212, %alloc_82[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            affine.yield %213 : f32
          }
          affine.for %arg5 = 0 to 128 {
            %210 = affine.load %alloc_82[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.divf %210, %209 : f32
            affine.store %211, %alloc_82[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_83 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 64 {
            %208 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_82[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x128xf32>
              %210 = affine.load %alloc_78[%arg2, %arg3, %arg6, %arg5] : memref<1x12x128x64xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_83[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_84 = memref.alloc() {alignment = 16 : i64} : memref<1x128x12x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.apply #map11(%arg2, %arg3, %arg4)
          %209 = affine.apply #map12(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_84, %alloc_83, %c64_i64, %209, %208) : (memref<1x128x12x64xf32>, memref<1x12x128x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_85 = memref.reinterpret_cast %alloc_84 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x12x64xf32> to memref<128x768xf32>
    %alloc_86 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_85[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %54[%arg4, %arg3] : memref<768x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %55[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_86[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_87 = memref.reinterpret_cast %alloc_86 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_88 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_66[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_87[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_88[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %186 = memref.get_global @constant_323 : memref<1xf32>
    %alloc_89 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_90 = arith.constant 1 : index
    %c128_91 = arith.constant 128 : index
    %c1_92 = arith.constant 1 : index
    %c0_93 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_89[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_88[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_89[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_89[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_89[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_89[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_94 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_89[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_89[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_94[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_95 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_88[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_88[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_95[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_96 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_97 = arith.constant 1 : index
    %c128_98 = arith.constant 128 : index
    %c1_99 = arith.constant 1 : index
    %c0_100 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_96[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_95[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_96[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_96[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_96[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_96[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_101 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_96[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_94[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %186[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_101[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_102 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_88[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_89[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_102[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_103 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_102[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_101[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %56[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %57[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_103[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_104 = memref.reinterpret_cast %alloc_103 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_105 = memref.alloc() {alignment = 128 : i64} : memref<128x3072xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 3072 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_104[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %58[%arg4, %arg3] : memref<768x3072xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %59[%arg3] : memref<3072xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_105[%arg2, %arg3] : memref<128x3072xf32>
      }
    }
    %reinterpret_cast_106 = memref.reinterpret_cast %alloc_105 to offset: [0], sizes: [1, 128, 3072], strides: [393216, 3072, 1] : memref<128x3072xf32> to memref<1x128x3072xf32>
    %alloc_107 = memref.alloc() {alignment = 16 : i64} : memref<1x128x3072xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 3072 {
          %208 = affine.load %reinterpret_cast_106[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
          %209 = arith.mulf %208, %cst_3 : f32
          %210 = math.powf %208, %cst_2 : f32
          %211 = arith.mulf %210, %cst_1 : f32
          %212 = arith.addf %208, %211 : f32
          %213 = arith.mulf %212, %cst_0 : f32
          %214 = math.tanh %213 : f32
          %215 = arith.addf %214, %cst_6 : f32
          %216 = arith.mulf %209, %215 : f32
          affine.store %216, %alloc_107[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
        }
      }
    }
    %reinterpret_cast_108 = memref.reinterpret_cast %alloc_107 to offset: [0], sizes: [128, 3072], strides: [3072, 1] : memref<1x128x3072xf32> to memref<128x3072xf32>
    %alloc_109 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 3072 {
          %211 = affine.load %reinterpret_cast_108[%arg2, %arg4] : memref<128x3072xf32>
          %212 = affine.load %60[%arg4, %arg3] : memref<3072x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %61[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_109[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_110 = memref.reinterpret_cast %alloc_109 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_111 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_103[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_110[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_111[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %187 = memref.get_global @constant_325 : memref<1xf32>
    %alloc_112 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_113 = arith.constant 1 : index
    %c128_114 = arith.constant 128 : index
    %c1_115 = arith.constant 1 : index
    %c0_116 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_112[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_111[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_112[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_112[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_112[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_112[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_117 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_112[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_112[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_117[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_118 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_111[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_111[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_118[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_119 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_120 = arith.constant 1 : index
    %c128_121 = arith.constant 128 : index
    %c1_122 = arith.constant 1 : index
    %c0_123 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_119[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_118[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_119[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_119[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_119[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_119[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_124 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_119[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_117[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %187[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_124[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_125 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_111[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_112[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_125[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_126 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_125[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_124[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %62[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %63[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_126[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_127 = memref.reinterpret_cast %alloc_126 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_128 = memref.alloc() {alignment = 128 : i64} : memref<128x2304xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 2304 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_127[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %64[%arg4, %arg3] : memref<768x2304xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %65[%arg3] : memref<2304xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_128[%arg2, %arg3] : memref<128x2304xf32>
      }
    }
    %reinterpret_cast_129 = memref.reinterpret_cast %alloc_128 to offset: [0], sizes: [1, 128, 2304], strides: [294912, 2304, 1] : memref<128x2304xf32> to memref<1x128x2304xf32>
    %alloc_130 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_131 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_132 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %reinterpret_cast_129[%arg2, %arg3, %arg4] : memref<1x128x2304xf32>
          affine.store %208, %alloc_130[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map(%arg4)
          %209 = affine.load %reinterpret_cast_129[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_131[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map1(%arg4)
          %209 = affine.load %reinterpret_cast_129[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_132[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_133 = memref.reinterpret_cast %alloc_130 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_134 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_134, %reinterpret_cast_133, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_135 = memref.reinterpret_cast %alloc_131 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_136 = memref.alloc() {alignment = 16 : i64} : memref<1x12x64x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 64 {
          affine.for %arg5 = 0 to 128 step 8 {
            %208 = affine.load %reinterpret_cast_135[%arg2, %arg5, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %208, %alloc_136[%arg2, %arg3, %arg4, %arg5] : memref<1x12x64x128xf32>
            %209 = affine.apply #map4(%arg5)
            %210 = affine.load %reinterpret_cast_135[%arg2, %209, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %210, %alloc_136[%arg2, %arg3, %arg4, %209] : memref<1x12x64x128xf32>
            %211 = affine.apply #map5(%arg5)
            %212 = affine.load %reinterpret_cast_135[%arg2, %211, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %212, %alloc_136[%arg2, %arg3, %arg4, %211] : memref<1x12x64x128xf32>
            %213 = affine.apply #map6(%arg5)
            %214 = affine.load %reinterpret_cast_135[%arg2, %213, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %214, %alloc_136[%arg2, %arg3, %arg4, %213] : memref<1x12x64x128xf32>
            %215 = affine.apply #map7(%arg5)
            %216 = affine.load %reinterpret_cast_135[%arg2, %215, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %216, %alloc_136[%arg2, %arg3, %arg4, %215] : memref<1x12x64x128xf32>
            %217 = affine.apply #map8(%arg5)
            %218 = affine.load %reinterpret_cast_135[%arg2, %217, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %218, %alloc_136[%arg2, %arg3, %arg4, %217] : memref<1x12x64x128xf32>
            %219 = affine.apply #map9(%arg5)
            %220 = affine.load %reinterpret_cast_135[%arg2, %219, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %220, %alloc_136[%arg2, %arg3, %arg4, %219] : memref<1x12x64x128xf32>
            %221 = affine.apply #map10(%arg5)
            %222 = affine.load %reinterpret_cast_135[%arg2, %221, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %222, %alloc_136[%arg2, %arg3, %arg4, %221] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %reinterpret_cast_137 = memref.reinterpret_cast %alloc_132 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_138 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_138, %reinterpret_cast_137, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %alloc_139 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_134[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x64xf32>
              %210 = affine.load %alloc_136[%arg2, %arg3, %arg6, %arg5] : memref<1x12x64x128xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_139[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_140 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_139[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %30[] : memref<f32>
            %210 = arith.divf %208, %209 : f32
            %211 = affine.load %29[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.mulf %210, %211 : f32
            affine.store %212, %alloc_140[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_141 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_140[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %alloc[%c0, %c0, %c0, %arg5] : memref<1x1x1x128xf32>
            %210 = arith.addf %208, %209 : f32
            %211 = affine.load %28[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.addf %210, %211 : f32
            affine.store %212, %alloc_141[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_142 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_4) -> (f32) {
            %210 = affine.load %alloc_141[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.maxf %arg6, %210 : f32
            affine.yield %211 : f32
          }
          %209 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_5) -> (f32) {
            %210 = affine.load %alloc_141[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.subf %210, %208 : f32
            %212 = math.exp %211 : f32
            %213 = arith.addf %arg6, %212 : f32
            affine.store %212, %alloc_142[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            affine.yield %213 : f32
          }
          affine.for %arg5 = 0 to 128 {
            %210 = affine.load %alloc_142[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.divf %210, %209 : f32
            affine.store %211, %alloc_142[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_143 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 64 {
            %208 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_142[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x128xf32>
              %210 = affine.load %alloc_138[%arg2, %arg3, %arg6, %arg5] : memref<1x12x128x64xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_143[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_144 = memref.alloc() {alignment = 16 : i64} : memref<1x128x12x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.apply #map11(%arg2, %arg3, %arg4)
          %209 = affine.apply #map12(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_144, %alloc_143, %c64_i64, %209, %208) : (memref<1x128x12x64xf32>, memref<1x12x128x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_145 = memref.reinterpret_cast %alloc_144 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x12x64xf32> to memref<128x768xf32>
    %alloc_146 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_145[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %66[%arg4, %arg3] : memref<768x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %67[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_146[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_147 = memref.reinterpret_cast %alloc_146 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_148 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_126[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_147[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_148[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %188 = memref.get_global @constant_327 : memref<1xf32>
    %alloc_149 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_150 = arith.constant 1 : index
    %c128_151 = arith.constant 128 : index
    %c1_152 = arith.constant 1 : index
    %c0_153 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_149[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_148[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_149[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_149[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_149[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_149[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_154 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_149[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_149[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_154[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_155 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_148[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_148[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_155[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_156 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_157 = arith.constant 1 : index
    %c128_158 = arith.constant 128 : index
    %c1_159 = arith.constant 1 : index
    %c0_160 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_156[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_155[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_156[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_156[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_156[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_156[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_161 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_156[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_154[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %188[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_161[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_162 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_148[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_149[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_162[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_163 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_162[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_161[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %68[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %69[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_163[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_164 = memref.reinterpret_cast %alloc_163 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_165 = memref.alloc() {alignment = 128 : i64} : memref<128x3072xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 3072 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_164[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %70[%arg4, %arg3] : memref<768x3072xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %71[%arg3] : memref<3072xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_165[%arg2, %arg3] : memref<128x3072xf32>
      }
    }
    %reinterpret_cast_166 = memref.reinterpret_cast %alloc_165 to offset: [0], sizes: [1, 128, 3072], strides: [393216, 3072, 1] : memref<128x3072xf32> to memref<1x128x3072xf32>
    %alloc_167 = memref.alloc() {alignment = 16 : i64} : memref<1x128x3072xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 3072 {
          %208 = affine.load %reinterpret_cast_166[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
          %209 = arith.mulf %208, %cst_3 : f32
          %210 = math.powf %208, %cst_2 : f32
          %211 = arith.mulf %210, %cst_1 : f32
          %212 = arith.addf %208, %211 : f32
          %213 = arith.mulf %212, %cst_0 : f32
          %214 = math.tanh %213 : f32
          %215 = arith.addf %214, %cst_6 : f32
          %216 = arith.mulf %209, %215 : f32
          affine.store %216, %alloc_167[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
        }
      }
    }
    %reinterpret_cast_168 = memref.reinterpret_cast %alloc_167 to offset: [0], sizes: [128, 3072], strides: [3072, 1] : memref<1x128x3072xf32> to memref<128x3072xf32>
    %alloc_169 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 3072 {
          %211 = affine.load %reinterpret_cast_168[%arg2, %arg4] : memref<128x3072xf32>
          %212 = affine.load %72[%arg4, %arg3] : memref<3072x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %73[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_169[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_170 = memref.reinterpret_cast %alloc_169 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_171 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_163[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_170[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_171[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %189 = memref.get_global @constant_329 : memref<1xf32>
    %alloc_172 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_173 = arith.constant 1 : index
    %c128_174 = arith.constant 128 : index
    %c1_175 = arith.constant 1 : index
    %c0_176 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_172[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_171[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_172[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_172[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_172[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_172[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_177 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_172[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_172[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_177[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_178 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_171[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_171[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_178[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_179 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_180 = arith.constant 1 : index
    %c128_181 = arith.constant 128 : index
    %c1_182 = arith.constant 1 : index
    %c0_183 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_179[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_178[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_179[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_179[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_179[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_179[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_184 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_179[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_177[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %189[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_184[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_185 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_171[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_172[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_185[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_186 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_185[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_184[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %74[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %75[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_186[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_187 = memref.reinterpret_cast %alloc_186 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_188 = memref.alloc() {alignment = 128 : i64} : memref<128x2304xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 2304 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_187[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %76[%arg4, %arg3] : memref<768x2304xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %77[%arg3] : memref<2304xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_188[%arg2, %arg3] : memref<128x2304xf32>
      }
    }
    %reinterpret_cast_189 = memref.reinterpret_cast %alloc_188 to offset: [0], sizes: [1, 128, 2304], strides: [294912, 2304, 1] : memref<128x2304xf32> to memref<1x128x2304xf32>
    %alloc_190 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_191 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_192 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %reinterpret_cast_189[%arg2, %arg3, %arg4] : memref<1x128x2304xf32>
          affine.store %208, %alloc_190[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map(%arg4)
          %209 = affine.load %reinterpret_cast_189[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_191[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map1(%arg4)
          %209 = affine.load %reinterpret_cast_189[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_192[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_193 = memref.reinterpret_cast %alloc_190 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_194 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_194, %reinterpret_cast_193, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_195 = memref.reinterpret_cast %alloc_191 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_196 = memref.alloc() {alignment = 16 : i64} : memref<1x12x64x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 64 {
          affine.for %arg5 = 0 to 128 step 8 {
            %208 = affine.load %reinterpret_cast_195[%arg2, %arg5, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %208, %alloc_196[%arg2, %arg3, %arg4, %arg5] : memref<1x12x64x128xf32>
            %209 = affine.apply #map4(%arg5)
            %210 = affine.load %reinterpret_cast_195[%arg2, %209, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %210, %alloc_196[%arg2, %arg3, %arg4, %209] : memref<1x12x64x128xf32>
            %211 = affine.apply #map5(%arg5)
            %212 = affine.load %reinterpret_cast_195[%arg2, %211, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %212, %alloc_196[%arg2, %arg3, %arg4, %211] : memref<1x12x64x128xf32>
            %213 = affine.apply #map6(%arg5)
            %214 = affine.load %reinterpret_cast_195[%arg2, %213, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %214, %alloc_196[%arg2, %arg3, %arg4, %213] : memref<1x12x64x128xf32>
            %215 = affine.apply #map7(%arg5)
            %216 = affine.load %reinterpret_cast_195[%arg2, %215, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %216, %alloc_196[%arg2, %arg3, %arg4, %215] : memref<1x12x64x128xf32>
            %217 = affine.apply #map8(%arg5)
            %218 = affine.load %reinterpret_cast_195[%arg2, %217, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %218, %alloc_196[%arg2, %arg3, %arg4, %217] : memref<1x12x64x128xf32>
            %219 = affine.apply #map9(%arg5)
            %220 = affine.load %reinterpret_cast_195[%arg2, %219, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %220, %alloc_196[%arg2, %arg3, %arg4, %219] : memref<1x12x64x128xf32>
            %221 = affine.apply #map10(%arg5)
            %222 = affine.load %reinterpret_cast_195[%arg2, %221, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %222, %alloc_196[%arg2, %arg3, %arg4, %221] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %reinterpret_cast_197 = memref.reinterpret_cast %alloc_192 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_198 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_198, %reinterpret_cast_197, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %alloc_199 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_194[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x64xf32>
              %210 = affine.load %alloc_196[%arg2, %arg3, %arg6, %arg5] : memref<1x12x64x128xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_199[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_200 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_199[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %27[] : memref<f32>
            %210 = arith.divf %208, %209 : f32
            %211 = affine.load %26[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.mulf %210, %211 : f32
            affine.store %212, %alloc_200[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_201 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_200[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %alloc[%c0, %c0, %c0, %arg5] : memref<1x1x1x128xf32>
            %210 = arith.addf %208, %209 : f32
            %211 = affine.load %25[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.addf %210, %211 : f32
            affine.store %212, %alloc_201[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_202 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_4) -> (f32) {
            %210 = affine.load %alloc_201[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.maxf %arg6, %210 : f32
            affine.yield %211 : f32
          }
          %209 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_5) -> (f32) {
            %210 = affine.load %alloc_201[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.subf %210, %208 : f32
            %212 = math.exp %211 : f32
            %213 = arith.addf %arg6, %212 : f32
            affine.store %212, %alloc_202[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            affine.yield %213 : f32
          }
          affine.for %arg5 = 0 to 128 {
            %210 = affine.load %alloc_202[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.divf %210, %209 : f32
            affine.store %211, %alloc_202[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_203 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 64 {
            %208 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_202[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x128xf32>
              %210 = affine.load %alloc_198[%arg2, %arg3, %arg6, %arg5] : memref<1x12x128x64xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_203[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_204 = memref.alloc() {alignment = 16 : i64} : memref<1x128x12x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.apply #map11(%arg2, %arg3, %arg4)
          %209 = affine.apply #map12(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_204, %alloc_203, %c64_i64, %209, %208) : (memref<1x128x12x64xf32>, memref<1x12x128x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_205 = memref.reinterpret_cast %alloc_204 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x12x64xf32> to memref<128x768xf32>
    %alloc_206 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_205[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %78[%arg4, %arg3] : memref<768x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %79[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_206[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_207 = memref.reinterpret_cast %alloc_206 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_208 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_186[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_207[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_208[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %190 = memref.get_global @constant_331 : memref<1xf32>
    %alloc_209 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_210 = arith.constant 1 : index
    %c128_211 = arith.constant 128 : index
    %c1_212 = arith.constant 1 : index
    %c0_213 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_209[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_208[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_209[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_209[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_209[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_209[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_214 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_209[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_209[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_214[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_215 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_208[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_208[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_215[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_216 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_217 = arith.constant 1 : index
    %c128_218 = arith.constant 128 : index
    %c1_219 = arith.constant 1 : index
    %c0_220 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_216[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_215[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_216[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_216[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_216[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_216[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_221 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_216[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_214[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %190[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_221[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_222 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_208[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_209[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_222[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_223 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_222[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_221[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %80[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %81[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_223[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_224 = memref.reinterpret_cast %alloc_223 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_225 = memref.alloc() {alignment = 128 : i64} : memref<128x3072xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 3072 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_224[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %82[%arg4, %arg3] : memref<768x3072xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %83[%arg3] : memref<3072xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_225[%arg2, %arg3] : memref<128x3072xf32>
      }
    }
    %reinterpret_cast_226 = memref.reinterpret_cast %alloc_225 to offset: [0], sizes: [1, 128, 3072], strides: [393216, 3072, 1] : memref<128x3072xf32> to memref<1x128x3072xf32>
    %alloc_227 = memref.alloc() {alignment = 16 : i64} : memref<1x128x3072xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 3072 {
          %208 = affine.load %reinterpret_cast_226[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
          %209 = arith.mulf %208, %cst_3 : f32
          %210 = math.powf %208, %cst_2 : f32
          %211 = arith.mulf %210, %cst_1 : f32
          %212 = arith.addf %208, %211 : f32
          %213 = arith.mulf %212, %cst_0 : f32
          %214 = math.tanh %213 : f32
          %215 = arith.addf %214, %cst_6 : f32
          %216 = arith.mulf %209, %215 : f32
          affine.store %216, %alloc_227[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
        }
      }
    }
    %reinterpret_cast_228 = memref.reinterpret_cast %alloc_227 to offset: [0], sizes: [128, 3072], strides: [3072, 1] : memref<1x128x3072xf32> to memref<128x3072xf32>
    %alloc_229 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 3072 {
          %211 = affine.load %reinterpret_cast_228[%arg2, %arg4] : memref<128x3072xf32>
          %212 = affine.load %84[%arg4, %arg3] : memref<3072x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %85[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_229[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_230 = memref.reinterpret_cast %alloc_229 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_231 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_223[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_230[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_231[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %191 = memref.get_global @constant_333 : memref<1xf32>
    %alloc_232 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_233 = arith.constant 1 : index
    %c128_234 = arith.constant 128 : index
    %c1_235 = arith.constant 1 : index
    %c0_236 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_232[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_231[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_232[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_232[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_232[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_232[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_237 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_232[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_232[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_237[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_238 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_231[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_231[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_238[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_239 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_240 = arith.constant 1 : index
    %c128_241 = arith.constant 128 : index
    %c1_242 = arith.constant 1 : index
    %c0_243 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_239[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_238[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_239[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_239[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_239[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_239[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_244 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_239[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_237[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %191[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_244[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_245 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_231[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_232[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_245[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_246 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_245[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_244[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %86[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %87[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_246[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_247 = memref.reinterpret_cast %alloc_246 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_248 = memref.alloc() {alignment = 128 : i64} : memref<128x2304xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 2304 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_247[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %88[%arg4, %arg3] : memref<768x2304xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %89[%arg3] : memref<2304xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_248[%arg2, %arg3] : memref<128x2304xf32>
      }
    }
    %reinterpret_cast_249 = memref.reinterpret_cast %alloc_248 to offset: [0], sizes: [1, 128, 2304], strides: [294912, 2304, 1] : memref<128x2304xf32> to memref<1x128x2304xf32>
    %alloc_250 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_251 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_252 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %reinterpret_cast_249[%arg2, %arg3, %arg4] : memref<1x128x2304xf32>
          affine.store %208, %alloc_250[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map(%arg4)
          %209 = affine.load %reinterpret_cast_249[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_251[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map1(%arg4)
          %209 = affine.load %reinterpret_cast_249[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_252[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_253 = memref.reinterpret_cast %alloc_250 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_254 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_254, %reinterpret_cast_253, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_255 = memref.reinterpret_cast %alloc_251 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_256 = memref.alloc() {alignment = 16 : i64} : memref<1x12x64x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 64 {
          affine.for %arg5 = 0 to 128 step 8 {
            %208 = affine.load %reinterpret_cast_255[%arg2, %arg5, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %208, %alloc_256[%arg2, %arg3, %arg4, %arg5] : memref<1x12x64x128xf32>
            %209 = affine.apply #map4(%arg5)
            %210 = affine.load %reinterpret_cast_255[%arg2, %209, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %210, %alloc_256[%arg2, %arg3, %arg4, %209] : memref<1x12x64x128xf32>
            %211 = affine.apply #map5(%arg5)
            %212 = affine.load %reinterpret_cast_255[%arg2, %211, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %212, %alloc_256[%arg2, %arg3, %arg4, %211] : memref<1x12x64x128xf32>
            %213 = affine.apply #map6(%arg5)
            %214 = affine.load %reinterpret_cast_255[%arg2, %213, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %214, %alloc_256[%arg2, %arg3, %arg4, %213] : memref<1x12x64x128xf32>
            %215 = affine.apply #map7(%arg5)
            %216 = affine.load %reinterpret_cast_255[%arg2, %215, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %216, %alloc_256[%arg2, %arg3, %arg4, %215] : memref<1x12x64x128xf32>
            %217 = affine.apply #map8(%arg5)
            %218 = affine.load %reinterpret_cast_255[%arg2, %217, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %218, %alloc_256[%arg2, %arg3, %arg4, %217] : memref<1x12x64x128xf32>
            %219 = affine.apply #map9(%arg5)
            %220 = affine.load %reinterpret_cast_255[%arg2, %219, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %220, %alloc_256[%arg2, %arg3, %arg4, %219] : memref<1x12x64x128xf32>
            %221 = affine.apply #map10(%arg5)
            %222 = affine.load %reinterpret_cast_255[%arg2, %221, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %222, %alloc_256[%arg2, %arg3, %arg4, %221] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %reinterpret_cast_257 = memref.reinterpret_cast %alloc_252 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_258 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_258, %reinterpret_cast_257, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %alloc_259 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_254[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x64xf32>
              %210 = affine.load %alloc_256[%arg2, %arg3, %arg6, %arg5] : memref<1x12x64x128xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_259[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_260 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_259[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %24[] : memref<f32>
            %210 = arith.divf %208, %209 : f32
            %211 = affine.load %23[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.mulf %210, %211 : f32
            affine.store %212, %alloc_260[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_261 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_260[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %alloc[%c0, %c0, %c0, %arg5] : memref<1x1x1x128xf32>
            %210 = arith.addf %208, %209 : f32
            %211 = affine.load %22[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.addf %210, %211 : f32
            affine.store %212, %alloc_261[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_262 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_4) -> (f32) {
            %210 = affine.load %alloc_261[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.maxf %arg6, %210 : f32
            affine.yield %211 : f32
          }
          %209 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_5) -> (f32) {
            %210 = affine.load %alloc_261[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.subf %210, %208 : f32
            %212 = math.exp %211 : f32
            %213 = arith.addf %arg6, %212 : f32
            affine.store %212, %alloc_262[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            affine.yield %213 : f32
          }
          affine.for %arg5 = 0 to 128 {
            %210 = affine.load %alloc_262[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.divf %210, %209 : f32
            affine.store %211, %alloc_262[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_263 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 64 {
            %208 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_262[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x128xf32>
              %210 = affine.load %alloc_258[%arg2, %arg3, %arg6, %arg5] : memref<1x12x128x64xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_263[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_264 = memref.alloc() {alignment = 16 : i64} : memref<1x128x12x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.apply #map11(%arg2, %arg3, %arg4)
          %209 = affine.apply #map12(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_264, %alloc_263, %c64_i64, %209, %208) : (memref<1x128x12x64xf32>, memref<1x12x128x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_265 = memref.reinterpret_cast %alloc_264 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x12x64xf32> to memref<128x768xf32>
    %alloc_266 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_265[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %90[%arg4, %arg3] : memref<768x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %91[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_266[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_267 = memref.reinterpret_cast %alloc_266 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_268 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_246[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_267[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_268[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %192 = memref.get_global @constant_335 : memref<1xf32>
    %alloc_269 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_270 = arith.constant 1 : index
    %c128_271 = arith.constant 128 : index
    %c1_272 = arith.constant 1 : index
    %c0_273 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_269[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_268[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_269[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_269[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_269[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_269[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_274 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_269[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_269[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_274[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_275 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_268[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_268[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_275[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_276 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_277 = arith.constant 1 : index
    %c128_278 = arith.constant 128 : index
    %c1_279 = arith.constant 1 : index
    %c0_280 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_276[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_275[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_276[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_276[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_276[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_276[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_281 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_276[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_274[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %192[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_281[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_282 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_268[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_269[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_282[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_283 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_282[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_281[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %92[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %93[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_283[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_284 = memref.reinterpret_cast %alloc_283 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_285 = memref.alloc() {alignment = 128 : i64} : memref<128x3072xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 3072 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_284[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %94[%arg4, %arg3] : memref<768x3072xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %95[%arg3] : memref<3072xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_285[%arg2, %arg3] : memref<128x3072xf32>
      }
    }
    %reinterpret_cast_286 = memref.reinterpret_cast %alloc_285 to offset: [0], sizes: [1, 128, 3072], strides: [393216, 3072, 1] : memref<128x3072xf32> to memref<1x128x3072xf32>
    %alloc_287 = memref.alloc() {alignment = 16 : i64} : memref<1x128x3072xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 3072 {
          %208 = affine.load %reinterpret_cast_286[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
          %209 = arith.mulf %208, %cst_3 : f32
          %210 = math.powf %208, %cst_2 : f32
          %211 = arith.mulf %210, %cst_1 : f32
          %212 = arith.addf %208, %211 : f32
          %213 = arith.mulf %212, %cst_0 : f32
          %214 = math.tanh %213 : f32
          %215 = arith.addf %214, %cst_6 : f32
          %216 = arith.mulf %209, %215 : f32
          affine.store %216, %alloc_287[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
        }
      }
    }
    %reinterpret_cast_288 = memref.reinterpret_cast %alloc_287 to offset: [0], sizes: [128, 3072], strides: [3072, 1] : memref<1x128x3072xf32> to memref<128x3072xf32>
    %alloc_289 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 3072 {
          %211 = affine.load %reinterpret_cast_288[%arg2, %arg4] : memref<128x3072xf32>
          %212 = affine.load %96[%arg4, %arg3] : memref<3072x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %97[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_289[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_290 = memref.reinterpret_cast %alloc_289 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_291 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_283[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_290[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_291[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %193 = memref.get_global @constant_337 : memref<1xf32>
    %alloc_292 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_293 = arith.constant 1 : index
    %c128_294 = arith.constant 128 : index
    %c1_295 = arith.constant 1 : index
    %c0_296 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_292[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_291[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_292[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_292[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_292[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_292[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_297 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_292[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_292[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_297[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_298 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_291[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_291[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_298[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_299 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_300 = arith.constant 1 : index
    %c128_301 = arith.constant 128 : index
    %c1_302 = arith.constant 1 : index
    %c0_303 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_299[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_298[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_299[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_299[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_299[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_299[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_304 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_299[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_297[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %193[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_304[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_305 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_291[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_292[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_305[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_306 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_305[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_304[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %98[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %99[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_306[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_307 = memref.reinterpret_cast %alloc_306 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_308 = memref.alloc() {alignment = 128 : i64} : memref<128x2304xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 2304 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_307[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %100[%arg4, %arg3] : memref<768x2304xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %101[%arg3] : memref<2304xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_308[%arg2, %arg3] : memref<128x2304xf32>
      }
    }
    %reinterpret_cast_309 = memref.reinterpret_cast %alloc_308 to offset: [0], sizes: [1, 128, 2304], strides: [294912, 2304, 1] : memref<128x2304xf32> to memref<1x128x2304xf32>
    %alloc_310 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_311 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_312 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %reinterpret_cast_309[%arg2, %arg3, %arg4] : memref<1x128x2304xf32>
          affine.store %208, %alloc_310[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map(%arg4)
          %209 = affine.load %reinterpret_cast_309[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_311[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map1(%arg4)
          %209 = affine.load %reinterpret_cast_309[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_312[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_313 = memref.reinterpret_cast %alloc_310 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_314 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_314, %reinterpret_cast_313, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_315 = memref.reinterpret_cast %alloc_311 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_316 = memref.alloc() {alignment = 16 : i64} : memref<1x12x64x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 64 {
          affine.for %arg5 = 0 to 128 step 8 {
            %208 = affine.load %reinterpret_cast_315[%arg2, %arg5, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %208, %alloc_316[%arg2, %arg3, %arg4, %arg5] : memref<1x12x64x128xf32>
            %209 = affine.apply #map4(%arg5)
            %210 = affine.load %reinterpret_cast_315[%arg2, %209, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %210, %alloc_316[%arg2, %arg3, %arg4, %209] : memref<1x12x64x128xf32>
            %211 = affine.apply #map5(%arg5)
            %212 = affine.load %reinterpret_cast_315[%arg2, %211, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %212, %alloc_316[%arg2, %arg3, %arg4, %211] : memref<1x12x64x128xf32>
            %213 = affine.apply #map6(%arg5)
            %214 = affine.load %reinterpret_cast_315[%arg2, %213, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %214, %alloc_316[%arg2, %arg3, %arg4, %213] : memref<1x12x64x128xf32>
            %215 = affine.apply #map7(%arg5)
            %216 = affine.load %reinterpret_cast_315[%arg2, %215, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %216, %alloc_316[%arg2, %arg3, %arg4, %215] : memref<1x12x64x128xf32>
            %217 = affine.apply #map8(%arg5)
            %218 = affine.load %reinterpret_cast_315[%arg2, %217, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %218, %alloc_316[%arg2, %arg3, %arg4, %217] : memref<1x12x64x128xf32>
            %219 = affine.apply #map9(%arg5)
            %220 = affine.load %reinterpret_cast_315[%arg2, %219, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %220, %alloc_316[%arg2, %arg3, %arg4, %219] : memref<1x12x64x128xf32>
            %221 = affine.apply #map10(%arg5)
            %222 = affine.load %reinterpret_cast_315[%arg2, %221, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %222, %alloc_316[%arg2, %arg3, %arg4, %221] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %reinterpret_cast_317 = memref.reinterpret_cast %alloc_312 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_318 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_318, %reinterpret_cast_317, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %alloc_319 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_314[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x64xf32>
              %210 = affine.load %alloc_316[%arg2, %arg3, %arg6, %arg5] : memref<1x12x64x128xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_319[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_320 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_319[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %21[] : memref<f32>
            %210 = arith.divf %208, %209 : f32
            %211 = affine.load %20[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.mulf %210, %211 : f32
            affine.store %212, %alloc_320[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_321 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_320[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %alloc[%c0, %c0, %c0, %arg5] : memref<1x1x1x128xf32>
            %210 = arith.addf %208, %209 : f32
            %211 = affine.load %19[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.addf %210, %211 : f32
            affine.store %212, %alloc_321[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_322 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_4) -> (f32) {
            %210 = affine.load %alloc_321[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.maxf %arg6, %210 : f32
            affine.yield %211 : f32
          }
          %209 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_5) -> (f32) {
            %210 = affine.load %alloc_321[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.subf %210, %208 : f32
            %212 = math.exp %211 : f32
            %213 = arith.addf %arg6, %212 : f32
            affine.store %212, %alloc_322[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            affine.yield %213 : f32
          }
          affine.for %arg5 = 0 to 128 {
            %210 = affine.load %alloc_322[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.divf %210, %209 : f32
            affine.store %211, %alloc_322[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_323 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 64 {
            %208 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_322[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x128xf32>
              %210 = affine.load %alloc_318[%arg2, %arg3, %arg6, %arg5] : memref<1x12x128x64xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_323[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_324 = memref.alloc() {alignment = 16 : i64} : memref<1x128x12x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.apply #map11(%arg2, %arg3, %arg4)
          %209 = affine.apply #map12(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_324, %alloc_323, %c64_i64, %209, %208) : (memref<1x128x12x64xf32>, memref<1x12x128x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_325 = memref.reinterpret_cast %alloc_324 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x12x64xf32> to memref<128x768xf32>
    %alloc_326 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_325[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %102[%arg4, %arg3] : memref<768x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %103[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_326[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_327 = memref.reinterpret_cast %alloc_326 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_328 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_306[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_327[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_328[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %194 = memref.get_global @constant_339 : memref<1xf32>
    %alloc_329 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_330 = arith.constant 1 : index
    %c128_331 = arith.constant 128 : index
    %c1_332 = arith.constant 1 : index
    %c0_333 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_329[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_328[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_329[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_329[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_329[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_329[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_334 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_329[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_329[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_334[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_335 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_328[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_328[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_335[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_336 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_337 = arith.constant 1 : index
    %c128_338 = arith.constant 128 : index
    %c1_339 = arith.constant 1 : index
    %c0_340 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_336[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_335[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_336[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_336[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_336[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_336[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_341 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_336[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_334[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %194[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_341[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_342 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_328[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_329[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_342[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_343 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_342[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_341[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %104[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %105[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_343[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_344 = memref.reinterpret_cast %alloc_343 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_345 = memref.alloc() {alignment = 128 : i64} : memref<128x3072xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 3072 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_344[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %106[%arg4, %arg3] : memref<768x3072xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %107[%arg3] : memref<3072xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_345[%arg2, %arg3] : memref<128x3072xf32>
      }
    }
    %reinterpret_cast_346 = memref.reinterpret_cast %alloc_345 to offset: [0], sizes: [1, 128, 3072], strides: [393216, 3072, 1] : memref<128x3072xf32> to memref<1x128x3072xf32>
    %alloc_347 = memref.alloc() {alignment = 16 : i64} : memref<1x128x3072xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 3072 {
          %208 = affine.load %reinterpret_cast_346[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
          %209 = arith.mulf %208, %cst_3 : f32
          %210 = math.powf %208, %cst_2 : f32
          %211 = arith.mulf %210, %cst_1 : f32
          %212 = arith.addf %208, %211 : f32
          %213 = arith.mulf %212, %cst_0 : f32
          %214 = math.tanh %213 : f32
          %215 = arith.addf %214, %cst_6 : f32
          %216 = arith.mulf %209, %215 : f32
          affine.store %216, %alloc_347[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
        }
      }
    }
    %reinterpret_cast_348 = memref.reinterpret_cast %alloc_347 to offset: [0], sizes: [128, 3072], strides: [3072, 1] : memref<1x128x3072xf32> to memref<128x3072xf32>
    %alloc_349 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 3072 {
          %211 = affine.load %reinterpret_cast_348[%arg2, %arg4] : memref<128x3072xf32>
          %212 = affine.load %108[%arg4, %arg3] : memref<3072x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %109[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_349[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_350 = memref.reinterpret_cast %alloc_349 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_351 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_343[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_350[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_351[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %195 = memref.get_global @constant_341 : memref<1xf32>
    %alloc_352 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_353 = arith.constant 1 : index
    %c128_354 = arith.constant 128 : index
    %c1_355 = arith.constant 1 : index
    %c0_356 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_352[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_351[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_352[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_352[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_352[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_352[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_357 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_352[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_352[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_357[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_358 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_351[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_351[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_358[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_359 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_360 = arith.constant 1 : index
    %c128_361 = arith.constant 128 : index
    %c1_362 = arith.constant 1 : index
    %c0_363 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_359[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_358[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_359[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_359[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_359[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_359[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_364 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_359[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_357[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %195[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_364[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_365 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_351[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_352[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_365[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_366 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_365[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_364[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %110[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %111[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_366[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_367 = memref.reinterpret_cast %alloc_366 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_368 = memref.alloc() {alignment = 128 : i64} : memref<128x2304xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 2304 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_367[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %112[%arg4, %arg3] : memref<768x2304xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %113[%arg3] : memref<2304xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_368[%arg2, %arg3] : memref<128x2304xf32>
      }
    }
    %reinterpret_cast_369 = memref.reinterpret_cast %alloc_368 to offset: [0], sizes: [1, 128, 2304], strides: [294912, 2304, 1] : memref<128x2304xf32> to memref<1x128x2304xf32>
    %alloc_370 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_371 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_372 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %reinterpret_cast_369[%arg2, %arg3, %arg4] : memref<1x128x2304xf32>
          affine.store %208, %alloc_370[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map(%arg4)
          %209 = affine.load %reinterpret_cast_369[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_371[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map1(%arg4)
          %209 = affine.load %reinterpret_cast_369[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_372[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_373 = memref.reinterpret_cast %alloc_370 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_374 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_374, %reinterpret_cast_373, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_375 = memref.reinterpret_cast %alloc_371 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_376 = memref.alloc() {alignment = 16 : i64} : memref<1x12x64x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 64 {
          affine.for %arg5 = 0 to 128 step 8 {
            %208 = affine.load %reinterpret_cast_375[%arg2, %arg5, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %208, %alloc_376[%arg2, %arg3, %arg4, %arg5] : memref<1x12x64x128xf32>
            %209 = affine.apply #map4(%arg5)
            %210 = affine.load %reinterpret_cast_375[%arg2, %209, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %210, %alloc_376[%arg2, %arg3, %arg4, %209] : memref<1x12x64x128xf32>
            %211 = affine.apply #map5(%arg5)
            %212 = affine.load %reinterpret_cast_375[%arg2, %211, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %212, %alloc_376[%arg2, %arg3, %arg4, %211] : memref<1x12x64x128xf32>
            %213 = affine.apply #map6(%arg5)
            %214 = affine.load %reinterpret_cast_375[%arg2, %213, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %214, %alloc_376[%arg2, %arg3, %arg4, %213] : memref<1x12x64x128xf32>
            %215 = affine.apply #map7(%arg5)
            %216 = affine.load %reinterpret_cast_375[%arg2, %215, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %216, %alloc_376[%arg2, %arg3, %arg4, %215] : memref<1x12x64x128xf32>
            %217 = affine.apply #map8(%arg5)
            %218 = affine.load %reinterpret_cast_375[%arg2, %217, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %218, %alloc_376[%arg2, %arg3, %arg4, %217] : memref<1x12x64x128xf32>
            %219 = affine.apply #map9(%arg5)
            %220 = affine.load %reinterpret_cast_375[%arg2, %219, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %220, %alloc_376[%arg2, %arg3, %arg4, %219] : memref<1x12x64x128xf32>
            %221 = affine.apply #map10(%arg5)
            %222 = affine.load %reinterpret_cast_375[%arg2, %221, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %222, %alloc_376[%arg2, %arg3, %arg4, %221] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %reinterpret_cast_377 = memref.reinterpret_cast %alloc_372 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_378 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_378, %reinterpret_cast_377, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %alloc_379 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_374[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x64xf32>
              %210 = affine.load %alloc_376[%arg2, %arg3, %arg6, %arg5] : memref<1x12x64x128xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_379[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_380 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_379[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %18[] : memref<f32>
            %210 = arith.divf %208, %209 : f32
            %211 = affine.load %17[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.mulf %210, %211 : f32
            affine.store %212, %alloc_380[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_381 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_380[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %alloc[%c0, %c0, %c0, %arg5] : memref<1x1x1x128xf32>
            %210 = arith.addf %208, %209 : f32
            %211 = affine.load %16[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.addf %210, %211 : f32
            affine.store %212, %alloc_381[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_382 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_4) -> (f32) {
            %210 = affine.load %alloc_381[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.maxf %arg6, %210 : f32
            affine.yield %211 : f32
          }
          %209 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_5) -> (f32) {
            %210 = affine.load %alloc_381[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.subf %210, %208 : f32
            %212 = math.exp %211 : f32
            %213 = arith.addf %arg6, %212 : f32
            affine.store %212, %alloc_382[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            affine.yield %213 : f32
          }
          affine.for %arg5 = 0 to 128 {
            %210 = affine.load %alloc_382[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.divf %210, %209 : f32
            affine.store %211, %alloc_382[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_383 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 64 {
            %208 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_382[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x128xf32>
              %210 = affine.load %alloc_378[%arg2, %arg3, %arg6, %arg5] : memref<1x12x128x64xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_383[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_384 = memref.alloc() {alignment = 16 : i64} : memref<1x128x12x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.apply #map11(%arg2, %arg3, %arg4)
          %209 = affine.apply #map12(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_384, %alloc_383, %c64_i64, %209, %208) : (memref<1x128x12x64xf32>, memref<1x12x128x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_385 = memref.reinterpret_cast %alloc_384 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x12x64xf32> to memref<128x768xf32>
    %alloc_386 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_385[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %114[%arg4, %arg3] : memref<768x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %115[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_386[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_387 = memref.reinterpret_cast %alloc_386 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_388 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_366[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_387[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_388[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %196 = memref.get_global @constant_343 : memref<1xf32>
    %alloc_389 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_390 = arith.constant 1 : index
    %c128_391 = arith.constant 128 : index
    %c1_392 = arith.constant 1 : index
    %c0_393 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_389[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_388[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_389[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_389[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_389[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_389[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_394 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_389[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_389[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_394[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_395 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_388[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_388[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_395[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_396 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_397 = arith.constant 1 : index
    %c128_398 = arith.constant 128 : index
    %c1_399 = arith.constant 1 : index
    %c0_400 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_396[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_395[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_396[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_396[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_396[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_396[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_401 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_396[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_394[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %196[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_401[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_402 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_388[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_389[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_402[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_403 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_402[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_401[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %116[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %117[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_403[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_404 = memref.reinterpret_cast %alloc_403 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_405 = memref.alloc() {alignment = 128 : i64} : memref<128x3072xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 3072 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_404[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %118[%arg4, %arg3] : memref<768x3072xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %119[%arg3] : memref<3072xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_405[%arg2, %arg3] : memref<128x3072xf32>
      }
    }
    %reinterpret_cast_406 = memref.reinterpret_cast %alloc_405 to offset: [0], sizes: [1, 128, 3072], strides: [393216, 3072, 1] : memref<128x3072xf32> to memref<1x128x3072xf32>
    %alloc_407 = memref.alloc() {alignment = 16 : i64} : memref<1x128x3072xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 3072 {
          %208 = affine.load %reinterpret_cast_406[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
          %209 = arith.mulf %208, %cst_3 : f32
          %210 = math.powf %208, %cst_2 : f32
          %211 = arith.mulf %210, %cst_1 : f32
          %212 = arith.addf %208, %211 : f32
          %213 = arith.mulf %212, %cst_0 : f32
          %214 = math.tanh %213 : f32
          %215 = arith.addf %214, %cst_6 : f32
          %216 = arith.mulf %209, %215 : f32
          affine.store %216, %alloc_407[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
        }
      }
    }
    %reinterpret_cast_408 = memref.reinterpret_cast %alloc_407 to offset: [0], sizes: [128, 3072], strides: [3072, 1] : memref<1x128x3072xf32> to memref<128x3072xf32>
    %alloc_409 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 3072 {
          %211 = affine.load %reinterpret_cast_408[%arg2, %arg4] : memref<128x3072xf32>
          %212 = affine.load %120[%arg4, %arg3] : memref<3072x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %121[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_409[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_410 = memref.reinterpret_cast %alloc_409 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_411 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_403[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_410[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_411[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %197 = memref.get_global @constant_345 : memref<1xf32>
    %alloc_412 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_413 = arith.constant 1 : index
    %c128_414 = arith.constant 128 : index
    %c1_415 = arith.constant 1 : index
    %c0_416 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_412[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_411[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_412[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_412[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_412[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_412[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_417 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_412[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_412[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_417[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_418 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_411[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_411[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_418[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_419 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_420 = arith.constant 1 : index
    %c128_421 = arith.constant 128 : index
    %c1_422 = arith.constant 1 : index
    %c0_423 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_419[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_418[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_419[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_419[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_419[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_419[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_424 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_419[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_417[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %197[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_424[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_425 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_411[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_412[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_425[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_426 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_425[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_424[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %122[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %123[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_426[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_427 = memref.reinterpret_cast %alloc_426 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_428 = memref.alloc() {alignment = 128 : i64} : memref<128x2304xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 2304 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_427[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %124[%arg4, %arg3] : memref<768x2304xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %125[%arg3] : memref<2304xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_428[%arg2, %arg3] : memref<128x2304xf32>
      }
    }
    %reinterpret_cast_429 = memref.reinterpret_cast %alloc_428 to offset: [0], sizes: [1, 128, 2304], strides: [294912, 2304, 1] : memref<128x2304xf32> to memref<1x128x2304xf32>
    %alloc_430 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_431 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_432 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %reinterpret_cast_429[%arg2, %arg3, %arg4] : memref<1x128x2304xf32>
          affine.store %208, %alloc_430[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map(%arg4)
          %209 = affine.load %reinterpret_cast_429[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_431[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map1(%arg4)
          %209 = affine.load %reinterpret_cast_429[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_432[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_433 = memref.reinterpret_cast %alloc_430 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_434 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_434, %reinterpret_cast_433, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_435 = memref.reinterpret_cast %alloc_431 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_436 = memref.alloc() {alignment = 16 : i64} : memref<1x12x64x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 64 {
          affine.for %arg5 = 0 to 128 step 8 {
            %208 = affine.load %reinterpret_cast_435[%arg2, %arg5, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %208, %alloc_436[%arg2, %arg3, %arg4, %arg5] : memref<1x12x64x128xf32>
            %209 = affine.apply #map4(%arg5)
            %210 = affine.load %reinterpret_cast_435[%arg2, %209, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %210, %alloc_436[%arg2, %arg3, %arg4, %209] : memref<1x12x64x128xf32>
            %211 = affine.apply #map5(%arg5)
            %212 = affine.load %reinterpret_cast_435[%arg2, %211, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %212, %alloc_436[%arg2, %arg3, %arg4, %211] : memref<1x12x64x128xf32>
            %213 = affine.apply #map6(%arg5)
            %214 = affine.load %reinterpret_cast_435[%arg2, %213, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %214, %alloc_436[%arg2, %arg3, %arg4, %213] : memref<1x12x64x128xf32>
            %215 = affine.apply #map7(%arg5)
            %216 = affine.load %reinterpret_cast_435[%arg2, %215, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %216, %alloc_436[%arg2, %arg3, %arg4, %215] : memref<1x12x64x128xf32>
            %217 = affine.apply #map8(%arg5)
            %218 = affine.load %reinterpret_cast_435[%arg2, %217, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %218, %alloc_436[%arg2, %arg3, %arg4, %217] : memref<1x12x64x128xf32>
            %219 = affine.apply #map9(%arg5)
            %220 = affine.load %reinterpret_cast_435[%arg2, %219, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %220, %alloc_436[%arg2, %arg3, %arg4, %219] : memref<1x12x64x128xf32>
            %221 = affine.apply #map10(%arg5)
            %222 = affine.load %reinterpret_cast_435[%arg2, %221, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %222, %alloc_436[%arg2, %arg3, %arg4, %221] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %reinterpret_cast_437 = memref.reinterpret_cast %alloc_432 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_438 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_438, %reinterpret_cast_437, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %alloc_439 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_434[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x64xf32>
              %210 = affine.load %alloc_436[%arg2, %arg3, %arg6, %arg5] : memref<1x12x64x128xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_439[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_440 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_439[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %15[] : memref<f32>
            %210 = arith.divf %208, %209 : f32
            %211 = affine.load %14[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.mulf %210, %211 : f32
            affine.store %212, %alloc_440[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_441 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_440[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %alloc[%c0, %c0, %c0, %arg5] : memref<1x1x1x128xf32>
            %210 = arith.addf %208, %209 : f32
            %211 = affine.load %13[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.addf %210, %211 : f32
            affine.store %212, %alloc_441[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_442 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_4) -> (f32) {
            %210 = affine.load %alloc_441[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.maxf %arg6, %210 : f32
            affine.yield %211 : f32
          }
          %209 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_5) -> (f32) {
            %210 = affine.load %alloc_441[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.subf %210, %208 : f32
            %212 = math.exp %211 : f32
            %213 = arith.addf %arg6, %212 : f32
            affine.store %212, %alloc_442[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            affine.yield %213 : f32
          }
          affine.for %arg5 = 0 to 128 {
            %210 = affine.load %alloc_442[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.divf %210, %209 : f32
            affine.store %211, %alloc_442[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_443 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 64 {
            %208 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_442[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x128xf32>
              %210 = affine.load %alloc_438[%arg2, %arg3, %arg6, %arg5] : memref<1x12x128x64xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_443[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_444 = memref.alloc() {alignment = 16 : i64} : memref<1x128x12x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.apply #map11(%arg2, %arg3, %arg4)
          %209 = affine.apply #map12(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_444, %alloc_443, %c64_i64, %209, %208) : (memref<1x128x12x64xf32>, memref<1x12x128x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_445 = memref.reinterpret_cast %alloc_444 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x12x64xf32> to memref<128x768xf32>
    %alloc_446 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_445[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %126[%arg4, %arg3] : memref<768x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %127[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_446[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_447 = memref.reinterpret_cast %alloc_446 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_448 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_426[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_447[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_448[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %198 = memref.get_global @constant_347 : memref<1xf32>
    %alloc_449 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_450 = arith.constant 1 : index
    %c128_451 = arith.constant 128 : index
    %c1_452 = arith.constant 1 : index
    %c0_453 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_449[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_448[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_449[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_449[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_449[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_449[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_454 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_449[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_449[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_454[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_455 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_448[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_448[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_455[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_456 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_457 = arith.constant 1 : index
    %c128_458 = arith.constant 128 : index
    %c1_459 = arith.constant 1 : index
    %c0_460 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_456[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_455[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_456[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_456[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_456[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_456[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_461 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_456[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_454[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %198[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_461[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_462 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_448[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_449[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_462[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_463 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_462[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_461[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %128[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %129[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_463[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_464 = memref.reinterpret_cast %alloc_463 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_465 = memref.alloc() {alignment = 128 : i64} : memref<128x3072xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 3072 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_464[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %130[%arg4, %arg3] : memref<768x3072xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %131[%arg3] : memref<3072xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_465[%arg2, %arg3] : memref<128x3072xf32>
      }
    }
    %reinterpret_cast_466 = memref.reinterpret_cast %alloc_465 to offset: [0], sizes: [1, 128, 3072], strides: [393216, 3072, 1] : memref<128x3072xf32> to memref<1x128x3072xf32>
    %alloc_467 = memref.alloc() {alignment = 16 : i64} : memref<1x128x3072xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 3072 {
          %208 = affine.load %reinterpret_cast_466[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
          %209 = arith.mulf %208, %cst_3 : f32
          %210 = math.powf %208, %cst_2 : f32
          %211 = arith.mulf %210, %cst_1 : f32
          %212 = arith.addf %208, %211 : f32
          %213 = arith.mulf %212, %cst_0 : f32
          %214 = math.tanh %213 : f32
          %215 = arith.addf %214, %cst_6 : f32
          %216 = arith.mulf %209, %215 : f32
          affine.store %216, %alloc_467[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
        }
      }
    }
    %reinterpret_cast_468 = memref.reinterpret_cast %alloc_467 to offset: [0], sizes: [128, 3072], strides: [3072, 1] : memref<1x128x3072xf32> to memref<128x3072xf32>
    %alloc_469 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 3072 {
          %211 = affine.load %reinterpret_cast_468[%arg2, %arg4] : memref<128x3072xf32>
          %212 = affine.load %132[%arg4, %arg3] : memref<3072x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %133[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_469[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_470 = memref.reinterpret_cast %alloc_469 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_471 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_463[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_470[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_471[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %199 = memref.get_global @constant_349 : memref<1xf32>
    %alloc_472 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_473 = arith.constant 1 : index
    %c128_474 = arith.constant 128 : index
    %c1_475 = arith.constant 1 : index
    %c0_476 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_472[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_471[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_472[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_472[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_472[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_472[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_477 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_472[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_472[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_477[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_478 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_471[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_471[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_478[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_479 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_480 = arith.constant 1 : index
    %c128_481 = arith.constant 128 : index
    %c1_482 = arith.constant 1 : index
    %c0_483 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_479[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_478[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_479[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_479[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_479[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_479[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_484 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_479[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_477[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %199[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_484[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_485 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_471[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_472[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_485[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_486 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_485[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_484[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %134[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %135[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_486[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_487 = memref.reinterpret_cast %alloc_486 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_488 = memref.alloc() {alignment = 128 : i64} : memref<128x2304xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 2304 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_487[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %136[%arg4, %arg3] : memref<768x2304xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %137[%arg3] : memref<2304xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_488[%arg2, %arg3] : memref<128x2304xf32>
      }
    }
    %reinterpret_cast_489 = memref.reinterpret_cast %alloc_488 to offset: [0], sizes: [1, 128, 2304], strides: [294912, 2304, 1] : memref<128x2304xf32> to memref<1x128x2304xf32>
    %alloc_490 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_491 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_492 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %reinterpret_cast_489[%arg2, %arg3, %arg4] : memref<1x128x2304xf32>
          affine.store %208, %alloc_490[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map(%arg4)
          %209 = affine.load %reinterpret_cast_489[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_491[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map1(%arg4)
          %209 = affine.load %reinterpret_cast_489[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_492[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_493 = memref.reinterpret_cast %alloc_490 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_494 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_494, %reinterpret_cast_493, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_495 = memref.reinterpret_cast %alloc_491 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_496 = memref.alloc() {alignment = 16 : i64} : memref<1x12x64x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 64 {
          affine.for %arg5 = 0 to 128 step 8 {
            %208 = affine.load %reinterpret_cast_495[%arg2, %arg5, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %208, %alloc_496[%arg2, %arg3, %arg4, %arg5] : memref<1x12x64x128xf32>
            %209 = affine.apply #map4(%arg5)
            %210 = affine.load %reinterpret_cast_495[%arg2, %209, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %210, %alloc_496[%arg2, %arg3, %arg4, %209] : memref<1x12x64x128xf32>
            %211 = affine.apply #map5(%arg5)
            %212 = affine.load %reinterpret_cast_495[%arg2, %211, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %212, %alloc_496[%arg2, %arg3, %arg4, %211] : memref<1x12x64x128xf32>
            %213 = affine.apply #map6(%arg5)
            %214 = affine.load %reinterpret_cast_495[%arg2, %213, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %214, %alloc_496[%arg2, %arg3, %arg4, %213] : memref<1x12x64x128xf32>
            %215 = affine.apply #map7(%arg5)
            %216 = affine.load %reinterpret_cast_495[%arg2, %215, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %216, %alloc_496[%arg2, %arg3, %arg4, %215] : memref<1x12x64x128xf32>
            %217 = affine.apply #map8(%arg5)
            %218 = affine.load %reinterpret_cast_495[%arg2, %217, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %218, %alloc_496[%arg2, %arg3, %arg4, %217] : memref<1x12x64x128xf32>
            %219 = affine.apply #map9(%arg5)
            %220 = affine.load %reinterpret_cast_495[%arg2, %219, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %220, %alloc_496[%arg2, %arg3, %arg4, %219] : memref<1x12x64x128xf32>
            %221 = affine.apply #map10(%arg5)
            %222 = affine.load %reinterpret_cast_495[%arg2, %221, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %222, %alloc_496[%arg2, %arg3, %arg4, %221] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %reinterpret_cast_497 = memref.reinterpret_cast %alloc_492 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_498 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_498, %reinterpret_cast_497, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %alloc_499 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_494[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x64xf32>
              %210 = affine.load %alloc_496[%arg2, %arg3, %arg6, %arg5] : memref<1x12x64x128xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_499[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_500 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_499[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %12[] : memref<f32>
            %210 = arith.divf %208, %209 : f32
            %211 = affine.load %11[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.mulf %210, %211 : f32
            affine.store %212, %alloc_500[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_501 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_500[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %alloc[%c0, %c0, %c0, %arg5] : memref<1x1x1x128xf32>
            %210 = arith.addf %208, %209 : f32
            %211 = affine.load %10[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.addf %210, %211 : f32
            affine.store %212, %alloc_501[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_502 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_4) -> (f32) {
            %210 = affine.load %alloc_501[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.maxf %arg6, %210 : f32
            affine.yield %211 : f32
          }
          %209 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_5) -> (f32) {
            %210 = affine.load %alloc_501[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.subf %210, %208 : f32
            %212 = math.exp %211 : f32
            %213 = arith.addf %arg6, %212 : f32
            affine.store %212, %alloc_502[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            affine.yield %213 : f32
          }
          affine.for %arg5 = 0 to 128 {
            %210 = affine.load %alloc_502[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.divf %210, %209 : f32
            affine.store %211, %alloc_502[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_503 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 64 {
            %208 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_502[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x128xf32>
              %210 = affine.load %alloc_498[%arg2, %arg3, %arg6, %arg5] : memref<1x12x128x64xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_503[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_504 = memref.alloc() {alignment = 16 : i64} : memref<1x128x12x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.apply #map11(%arg2, %arg3, %arg4)
          %209 = affine.apply #map12(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_504, %alloc_503, %c64_i64, %209, %208) : (memref<1x128x12x64xf32>, memref<1x12x128x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_505 = memref.reinterpret_cast %alloc_504 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x12x64xf32> to memref<128x768xf32>
    %alloc_506 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_505[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %138[%arg4, %arg3] : memref<768x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %139[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_506[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_507 = memref.reinterpret_cast %alloc_506 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_508 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_486[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_507[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_508[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %200 = memref.get_global @constant_351 : memref<1xf32>
    %alloc_509 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_510 = arith.constant 1 : index
    %c128_511 = arith.constant 128 : index
    %c1_512 = arith.constant 1 : index
    %c0_513 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_509[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_508[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_509[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_509[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_509[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_509[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_514 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_509[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_509[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_514[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_515 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_508[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_508[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_515[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_516 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_517 = arith.constant 1 : index
    %c128_518 = arith.constant 128 : index
    %c1_519 = arith.constant 1 : index
    %c0_520 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_516[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_515[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_516[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_516[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_516[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_516[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_521 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_516[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_514[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %200[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_521[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_522 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_508[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_509[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_522[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_523 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_522[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_521[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %140[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %141[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_523[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_524 = memref.reinterpret_cast %alloc_523 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_525 = memref.alloc() {alignment = 128 : i64} : memref<128x3072xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 3072 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_524[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %142[%arg4, %arg3] : memref<768x3072xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %143[%arg3] : memref<3072xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_525[%arg2, %arg3] : memref<128x3072xf32>
      }
    }
    %reinterpret_cast_526 = memref.reinterpret_cast %alloc_525 to offset: [0], sizes: [1, 128, 3072], strides: [393216, 3072, 1] : memref<128x3072xf32> to memref<1x128x3072xf32>
    %alloc_527 = memref.alloc() {alignment = 16 : i64} : memref<1x128x3072xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 3072 {
          %208 = affine.load %reinterpret_cast_526[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
          %209 = arith.mulf %208, %cst_3 : f32
          %210 = math.powf %208, %cst_2 : f32
          %211 = arith.mulf %210, %cst_1 : f32
          %212 = arith.addf %208, %211 : f32
          %213 = arith.mulf %212, %cst_0 : f32
          %214 = math.tanh %213 : f32
          %215 = arith.addf %214, %cst_6 : f32
          %216 = arith.mulf %209, %215 : f32
          affine.store %216, %alloc_527[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
        }
      }
    }
    %reinterpret_cast_528 = memref.reinterpret_cast %alloc_527 to offset: [0], sizes: [128, 3072], strides: [3072, 1] : memref<1x128x3072xf32> to memref<128x3072xf32>
    %alloc_529 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 3072 {
          %211 = affine.load %reinterpret_cast_528[%arg2, %arg4] : memref<128x3072xf32>
          %212 = affine.load %144[%arg4, %arg3] : memref<3072x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %145[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_529[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_530 = memref.reinterpret_cast %alloc_529 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_531 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_523[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_530[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_531[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %201 = memref.get_global @constant_353 : memref<1xf32>
    %alloc_532 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_533 = arith.constant 1 : index
    %c128_534 = arith.constant 128 : index
    %c1_535 = arith.constant 1 : index
    %c0_536 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_532[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_531[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_532[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_532[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_532[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_532[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_537 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_532[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_532[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_537[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_538 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_531[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_531[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_538[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_539 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_540 = arith.constant 1 : index
    %c128_541 = arith.constant 128 : index
    %c1_542 = arith.constant 1 : index
    %c0_543 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_539[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_538[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_539[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_539[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_539[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_539[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_544 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_539[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_537[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %201[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_544[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_545 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_531[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_532[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_545[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_546 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_545[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_544[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %146[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %147[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_546[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_547 = memref.reinterpret_cast %alloc_546 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_548 = memref.alloc() {alignment = 128 : i64} : memref<128x2304xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 2304 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_547[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %148[%arg4, %arg3] : memref<768x2304xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %149[%arg3] : memref<2304xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_548[%arg2, %arg3] : memref<128x2304xf32>
      }
    }
    %reinterpret_cast_549 = memref.reinterpret_cast %alloc_548 to offset: [0], sizes: [1, 128, 2304], strides: [294912, 2304, 1] : memref<128x2304xf32> to memref<1x128x2304xf32>
    %alloc_550 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_551 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_552 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %reinterpret_cast_549[%arg2, %arg3, %arg4] : memref<1x128x2304xf32>
          affine.store %208, %alloc_550[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map(%arg4)
          %209 = affine.load %reinterpret_cast_549[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_551[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map1(%arg4)
          %209 = affine.load %reinterpret_cast_549[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_552[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_553 = memref.reinterpret_cast %alloc_550 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_554 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_554, %reinterpret_cast_553, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_555 = memref.reinterpret_cast %alloc_551 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_556 = memref.alloc() {alignment = 16 : i64} : memref<1x12x64x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 64 {
          affine.for %arg5 = 0 to 128 step 8 {
            %208 = affine.load %reinterpret_cast_555[%arg2, %arg5, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %208, %alloc_556[%arg2, %arg3, %arg4, %arg5] : memref<1x12x64x128xf32>
            %209 = affine.apply #map4(%arg5)
            %210 = affine.load %reinterpret_cast_555[%arg2, %209, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %210, %alloc_556[%arg2, %arg3, %arg4, %209] : memref<1x12x64x128xf32>
            %211 = affine.apply #map5(%arg5)
            %212 = affine.load %reinterpret_cast_555[%arg2, %211, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %212, %alloc_556[%arg2, %arg3, %arg4, %211] : memref<1x12x64x128xf32>
            %213 = affine.apply #map6(%arg5)
            %214 = affine.load %reinterpret_cast_555[%arg2, %213, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %214, %alloc_556[%arg2, %arg3, %arg4, %213] : memref<1x12x64x128xf32>
            %215 = affine.apply #map7(%arg5)
            %216 = affine.load %reinterpret_cast_555[%arg2, %215, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %216, %alloc_556[%arg2, %arg3, %arg4, %215] : memref<1x12x64x128xf32>
            %217 = affine.apply #map8(%arg5)
            %218 = affine.load %reinterpret_cast_555[%arg2, %217, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %218, %alloc_556[%arg2, %arg3, %arg4, %217] : memref<1x12x64x128xf32>
            %219 = affine.apply #map9(%arg5)
            %220 = affine.load %reinterpret_cast_555[%arg2, %219, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %220, %alloc_556[%arg2, %arg3, %arg4, %219] : memref<1x12x64x128xf32>
            %221 = affine.apply #map10(%arg5)
            %222 = affine.load %reinterpret_cast_555[%arg2, %221, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %222, %alloc_556[%arg2, %arg3, %arg4, %221] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %reinterpret_cast_557 = memref.reinterpret_cast %alloc_552 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_558 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_558, %reinterpret_cast_557, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %alloc_559 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_554[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x64xf32>
              %210 = affine.load %alloc_556[%arg2, %arg3, %arg6, %arg5] : memref<1x12x64x128xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_559[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_560 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_559[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %9[] : memref<f32>
            %210 = arith.divf %208, %209 : f32
            %211 = affine.load %8[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.mulf %210, %211 : f32
            affine.store %212, %alloc_560[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_561 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_560[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %alloc[%c0, %c0, %c0, %arg5] : memref<1x1x1x128xf32>
            %210 = arith.addf %208, %209 : f32
            %211 = affine.load %7[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.addf %210, %211 : f32
            affine.store %212, %alloc_561[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_562 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_4) -> (f32) {
            %210 = affine.load %alloc_561[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.maxf %arg6, %210 : f32
            affine.yield %211 : f32
          }
          %209 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_5) -> (f32) {
            %210 = affine.load %alloc_561[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.subf %210, %208 : f32
            %212 = math.exp %211 : f32
            %213 = arith.addf %arg6, %212 : f32
            affine.store %212, %alloc_562[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            affine.yield %213 : f32
          }
          affine.for %arg5 = 0 to 128 {
            %210 = affine.load %alloc_562[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.divf %210, %209 : f32
            affine.store %211, %alloc_562[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_563 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 64 {
            %208 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_562[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x128xf32>
              %210 = affine.load %alloc_558[%arg2, %arg3, %arg6, %arg5] : memref<1x12x128x64xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_563[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_564 = memref.alloc() {alignment = 16 : i64} : memref<1x128x12x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.apply #map11(%arg2, %arg3, %arg4)
          %209 = affine.apply #map12(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_564, %alloc_563, %c64_i64, %209, %208) : (memref<1x128x12x64xf32>, memref<1x12x128x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_565 = memref.reinterpret_cast %alloc_564 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x12x64xf32> to memref<128x768xf32>
    %alloc_566 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_565[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %150[%arg4, %arg3] : memref<768x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %151[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_566[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_567 = memref.reinterpret_cast %alloc_566 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_568 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_546[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_567[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_568[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %202 = memref.get_global @constant_355 : memref<1xf32>
    %alloc_569 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_570 = arith.constant 1 : index
    %c128_571 = arith.constant 128 : index
    %c1_572 = arith.constant 1 : index
    %c0_573 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_569[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_568[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_569[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_569[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_569[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_569[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_574 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_569[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_569[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_574[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_575 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_568[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_568[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_575[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_576 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_577 = arith.constant 1 : index
    %c128_578 = arith.constant 128 : index
    %c1_579 = arith.constant 1 : index
    %c0_580 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_576[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_575[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_576[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_576[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_576[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_576[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_581 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_576[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_574[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %202[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_581[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_582 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_568[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_569[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_582[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_583 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_582[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_581[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %152[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %153[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_583[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_584 = memref.reinterpret_cast %alloc_583 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_585 = memref.alloc() {alignment = 128 : i64} : memref<128x3072xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 3072 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_584[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %154[%arg4, %arg3] : memref<768x3072xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %155[%arg3] : memref<3072xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_585[%arg2, %arg3] : memref<128x3072xf32>
      }
    }
    %reinterpret_cast_586 = memref.reinterpret_cast %alloc_585 to offset: [0], sizes: [1, 128, 3072], strides: [393216, 3072, 1] : memref<128x3072xf32> to memref<1x128x3072xf32>
    %alloc_587 = memref.alloc() {alignment = 16 : i64} : memref<1x128x3072xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 3072 {
          %208 = affine.load %reinterpret_cast_586[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
          %209 = arith.mulf %208, %cst_3 : f32
          %210 = math.powf %208, %cst_2 : f32
          %211 = arith.mulf %210, %cst_1 : f32
          %212 = arith.addf %208, %211 : f32
          %213 = arith.mulf %212, %cst_0 : f32
          %214 = math.tanh %213 : f32
          %215 = arith.addf %214, %cst_6 : f32
          %216 = arith.mulf %209, %215 : f32
          affine.store %216, %alloc_587[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
        }
      }
    }
    %reinterpret_cast_588 = memref.reinterpret_cast %alloc_587 to offset: [0], sizes: [128, 3072], strides: [3072, 1] : memref<1x128x3072xf32> to memref<128x3072xf32>
    %alloc_589 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 3072 {
          %211 = affine.load %reinterpret_cast_588[%arg2, %arg4] : memref<128x3072xf32>
          %212 = affine.load %156[%arg4, %arg3] : memref<3072x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %157[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_589[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_590 = memref.reinterpret_cast %alloc_589 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_591 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_583[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_590[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_591[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %203 = memref.get_global @constant_357 : memref<1xf32>
    %alloc_592 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_593 = arith.constant 1 : index
    %c128_594 = arith.constant 128 : index
    %c1_595 = arith.constant 1 : index
    %c0_596 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_592[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_591[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_592[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_592[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_592[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_592[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_597 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_592[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_592[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_597[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_598 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_591[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_591[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_598[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_599 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_600 = arith.constant 1 : index
    %c128_601 = arith.constant 128 : index
    %c1_602 = arith.constant 1 : index
    %c0_603 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_599[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_598[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_599[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_599[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_599[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_599[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_604 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_599[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_597[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %203[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_604[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_605 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_591[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_592[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_605[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_606 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_605[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_604[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %158[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %159[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_606[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_607 = memref.reinterpret_cast %alloc_606 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_608 = memref.alloc() {alignment = 128 : i64} : memref<128x2304xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 2304 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_607[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %160[%arg4, %arg3] : memref<768x2304xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %161[%arg3] : memref<2304xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_608[%arg2, %arg3] : memref<128x2304xf32>
      }
    }
    %reinterpret_cast_609 = memref.reinterpret_cast %alloc_608 to offset: [0], sizes: [1, 128, 2304], strides: [294912, 2304, 1] : memref<128x2304xf32> to memref<1x128x2304xf32>
    %alloc_610 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_611 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_612 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %reinterpret_cast_609[%arg2, %arg3, %arg4] : memref<1x128x2304xf32>
          affine.store %208, %alloc_610[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map(%arg4)
          %209 = affine.load %reinterpret_cast_609[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_611[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map1(%arg4)
          %209 = affine.load %reinterpret_cast_609[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_612[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_613 = memref.reinterpret_cast %alloc_610 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_614 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_614, %reinterpret_cast_613, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_615 = memref.reinterpret_cast %alloc_611 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_616 = memref.alloc() {alignment = 16 : i64} : memref<1x12x64x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 64 {
          affine.for %arg5 = 0 to 128 step 8 {
            %208 = affine.load %reinterpret_cast_615[%arg2, %arg5, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %208, %alloc_616[%arg2, %arg3, %arg4, %arg5] : memref<1x12x64x128xf32>
            %209 = affine.apply #map4(%arg5)
            %210 = affine.load %reinterpret_cast_615[%arg2, %209, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %210, %alloc_616[%arg2, %arg3, %arg4, %209] : memref<1x12x64x128xf32>
            %211 = affine.apply #map5(%arg5)
            %212 = affine.load %reinterpret_cast_615[%arg2, %211, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %212, %alloc_616[%arg2, %arg3, %arg4, %211] : memref<1x12x64x128xf32>
            %213 = affine.apply #map6(%arg5)
            %214 = affine.load %reinterpret_cast_615[%arg2, %213, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %214, %alloc_616[%arg2, %arg3, %arg4, %213] : memref<1x12x64x128xf32>
            %215 = affine.apply #map7(%arg5)
            %216 = affine.load %reinterpret_cast_615[%arg2, %215, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %216, %alloc_616[%arg2, %arg3, %arg4, %215] : memref<1x12x64x128xf32>
            %217 = affine.apply #map8(%arg5)
            %218 = affine.load %reinterpret_cast_615[%arg2, %217, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %218, %alloc_616[%arg2, %arg3, %arg4, %217] : memref<1x12x64x128xf32>
            %219 = affine.apply #map9(%arg5)
            %220 = affine.load %reinterpret_cast_615[%arg2, %219, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %220, %alloc_616[%arg2, %arg3, %arg4, %219] : memref<1x12x64x128xf32>
            %221 = affine.apply #map10(%arg5)
            %222 = affine.load %reinterpret_cast_615[%arg2, %221, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %222, %alloc_616[%arg2, %arg3, %arg4, %221] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %reinterpret_cast_617 = memref.reinterpret_cast %alloc_612 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_618 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_618, %reinterpret_cast_617, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %alloc_619 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_614[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x64xf32>
              %210 = affine.load %alloc_616[%arg2, %arg3, %arg6, %arg5] : memref<1x12x64x128xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_619[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_620 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_619[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %6[] : memref<f32>
            %210 = arith.divf %208, %209 : f32
            %211 = affine.load %5[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.mulf %210, %211 : f32
            affine.store %212, %alloc_620[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_621 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_620[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %alloc[%c0, %c0, %c0, %arg5] : memref<1x1x1x128xf32>
            %210 = arith.addf %208, %209 : f32
            %211 = affine.load %4[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.addf %210, %211 : f32
            affine.store %212, %alloc_621[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_622 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_4) -> (f32) {
            %210 = affine.load %alloc_621[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.maxf %arg6, %210 : f32
            affine.yield %211 : f32
          }
          %209 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_5) -> (f32) {
            %210 = affine.load %alloc_621[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.subf %210, %208 : f32
            %212 = math.exp %211 : f32
            %213 = arith.addf %arg6, %212 : f32
            affine.store %212, %alloc_622[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            affine.yield %213 : f32
          }
          affine.for %arg5 = 0 to 128 {
            %210 = affine.load %alloc_622[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.divf %210, %209 : f32
            affine.store %211, %alloc_622[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_623 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 64 {
            %208 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_622[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x128xf32>
              %210 = affine.load %alloc_618[%arg2, %arg3, %arg6, %arg5] : memref<1x12x128x64xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_623[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_624 = memref.alloc() {alignment = 16 : i64} : memref<1x128x12x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.apply #map11(%arg2, %arg3, %arg4)
          %209 = affine.apply #map12(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_624, %alloc_623, %c64_i64, %209, %208) : (memref<1x128x12x64xf32>, memref<1x12x128x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_625 = memref.reinterpret_cast %alloc_624 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x12x64xf32> to memref<128x768xf32>
    %alloc_626 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_625[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %162[%arg4, %arg3] : memref<768x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %163[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_626[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_627 = memref.reinterpret_cast %alloc_626 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_628 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_606[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_627[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_628[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %204 = memref.get_global @constant_359 : memref<1xf32>
    %alloc_629 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_630 = arith.constant 1 : index
    %c128_631 = arith.constant 128 : index
    %c1_632 = arith.constant 1 : index
    %c0_633 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_629[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_628[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_629[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_629[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_629[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_629[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_634 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_629[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_629[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_634[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_635 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_628[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_628[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_635[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_636 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_637 = arith.constant 1 : index
    %c128_638 = arith.constant 128 : index
    %c1_639 = arith.constant 1 : index
    %c0_640 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_636[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_635[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_636[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_636[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_636[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_636[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_641 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_636[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_634[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %204[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_641[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_642 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_628[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_629[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_642[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_643 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_642[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_641[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %164[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %165[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_643[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_644 = memref.reinterpret_cast %alloc_643 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_645 = memref.alloc() {alignment = 128 : i64} : memref<128x3072xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 3072 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_644[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %166[%arg4, %arg3] : memref<768x3072xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %167[%arg3] : memref<3072xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_645[%arg2, %arg3] : memref<128x3072xf32>
      }
    }
    %reinterpret_cast_646 = memref.reinterpret_cast %alloc_645 to offset: [0], sizes: [1, 128, 3072], strides: [393216, 3072, 1] : memref<128x3072xf32> to memref<1x128x3072xf32>
    %alloc_647 = memref.alloc() {alignment = 16 : i64} : memref<1x128x3072xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 3072 {
          %208 = affine.load %reinterpret_cast_646[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
          %209 = arith.mulf %208, %cst_3 : f32
          %210 = math.powf %208, %cst_2 : f32
          %211 = arith.mulf %210, %cst_1 : f32
          %212 = arith.addf %208, %211 : f32
          %213 = arith.mulf %212, %cst_0 : f32
          %214 = math.tanh %213 : f32
          %215 = arith.addf %214, %cst_6 : f32
          %216 = arith.mulf %209, %215 : f32
          affine.store %216, %alloc_647[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
        }
      }
    }
    %reinterpret_cast_648 = memref.reinterpret_cast %alloc_647 to offset: [0], sizes: [128, 3072], strides: [3072, 1] : memref<1x128x3072xf32> to memref<128x3072xf32>
    %alloc_649 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 3072 {
          %211 = affine.load %reinterpret_cast_648[%arg2, %arg4] : memref<128x3072xf32>
          %212 = affine.load %168[%arg4, %arg3] : memref<3072x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %169[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_649[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_650 = memref.reinterpret_cast %alloc_649 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_651 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_643[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_650[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_651[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %205 = memref.get_global @constant_361 : memref<1xf32>
    %alloc_652 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_653 = arith.constant 1 : index
    %c128_654 = arith.constant 128 : index
    %c1_655 = arith.constant 1 : index
    %c0_656 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_652[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_651[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_652[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_652[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_652[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_652[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_657 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_652[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_652[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_657[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_658 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_651[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_651[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_658[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_659 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_660 = arith.constant 1 : index
    %c128_661 = arith.constant 128 : index
    %c1_662 = arith.constant 1 : index
    %c0_663 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_659[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_658[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_659[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_659[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_659[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_659[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_664 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_659[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_657[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %205[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_664[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_665 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_651[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_652[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_665[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_666 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_665[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_664[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %170[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %171[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_666[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_667 = memref.reinterpret_cast %alloc_666 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_668 = memref.alloc() {alignment = 128 : i64} : memref<128x2304xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 2304 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_667[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %172[%arg4, %arg3] : memref<768x2304xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %173[%arg3] : memref<2304xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_668[%arg2, %arg3] : memref<128x2304xf32>
      }
    }
    %reinterpret_cast_669 = memref.reinterpret_cast %alloc_668 to offset: [0], sizes: [1, 128, 2304], strides: [294912, 2304, 1] : memref<128x2304xf32> to memref<1x128x2304xf32>
    %alloc_670 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_671 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    %alloc_672 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %reinterpret_cast_669[%arg2, %arg3, %arg4] : memref<1x128x2304xf32>
          affine.store %208, %alloc_670[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map(%arg4)
          %209 = affine.load %reinterpret_cast_669[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_671[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.apply #map1(%arg4)
          %209 = affine.load %reinterpret_cast_669[%arg2, %arg3, %208] : memref<1x128x2304xf32>
          affine.store %209, %alloc_672[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_673 = memref.reinterpret_cast %alloc_670 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_674 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_674, %reinterpret_cast_673, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_675 = memref.reinterpret_cast %alloc_671 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_676 = memref.alloc() {alignment = 16 : i64} : memref<1x12x64x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 64 {
          affine.for %arg5 = 0 to 128 step 8 {
            %208 = affine.load %reinterpret_cast_675[%arg2, %arg5, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %208, %alloc_676[%arg2, %arg3, %arg4, %arg5] : memref<1x12x64x128xf32>
            %209 = affine.apply #map4(%arg5)
            %210 = affine.load %reinterpret_cast_675[%arg2, %209, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %210, %alloc_676[%arg2, %arg3, %arg4, %209] : memref<1x12x64x128xf32>
            %211 = affine.apply #map5(%arg5)
            %212 = affine.load %reinterpret_cast_675[%arg2, %211, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %212, %alloc_676[%arg2, %arg3, %arg4, %211] : memref<1x12x64x128xf32>
            %213 = affine.apply #map6(%arg5)
            %214 = affine.load %reinterpret_cast_675[%arg2, %213, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %214, %alloc_676[%arg2, %arg3, %arg4, %213] : memref<1x12x64x128xf32>
            %215 = affine.apply #map7(%arg5)
            %216 = affine.load %reinterpret_cast_675[%arg2, %215, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %216, %alloc_676[%arg2, %arg3, %arg4, %215] : memref<1x12x64x128xf32>
            %217 = affine.apply #map8(%arg5)
            %218 = affine.load %reinterpret_cast_675[%arg2, %217, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %218, %alloc_676[%arg2, %arg3, %arg4, %217] : memref<1x12x64x128xf32>
            %219 = affine.apply #map9(%arg5)
            %220 = affine.load %reinterpret_cast_675[%arg2, %219, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %220, %alloc_676[%arg2, %arg3, %arg4, %219] : memref<1x12x64x128xf32>
            %221 = affine.apply #map10(%arg5)
            %222 = affine.load %reinterpret_cast_675[%arg2, %221, %arg3, %arg4] : memref<1x128x12x64xf32>
            affine.store %222, %alloc_676[%arg2, %arg3, %arg4, %221] : memref<1x12x64x128xf32>
          }
        }
      }
    }
    %reinterpret_cast_677 = memref.reinterpret_cast %alloc_672 to offset: [0], sizes: [1, 128, 12, 64], strides: [98304, 768, 64, 1] : memref<1x128x768xf32> to memref<1x128x12x64xf32>
    %alloc_678 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 12 {
          %208 = affine.apply #map2(%arg2, %arg3, %arg4)
          %209 = affine.apply #map3(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_678, %reinterpret_cast_677, %c64_i64, %209, %208) : (memref<1x12x128x64xf32>, memref<1x128x12x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %alloc_679 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.for %arg6 = 0 to 64 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_674[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x64xf32>
              %210 = affine.load %alloc_676[%arg2, %arg3, %arg6, %arg5] : memref<1x12x64x128xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_679[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_680 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_679[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %3[] : memref<f32>
            %210 = arith.divf %208, %209 : f32
            %211 = affine.load %2[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.mulf %210, %211 : f32
            affine.store %212, %alloc_680[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_681 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 128 {
            %208 = affine.load %alloc_680[%c0, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %209 = affine.load %alloc[%c0, %c0, %c0, %arg5] : memref<1x1x1x128xf32>
            %210 = arith.addf %208, %209 : f32
            %211 = affine.load %1[%c0, %c0, %arg4, %arg5] : memref<1x1x128x128xf32>
            %212 = arith.addf %210, %211 : f32
            affine.store %212, %alloc_681[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_682 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x128xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_4) -> (f32) {
            %210 = affine.load %alloc_681[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.maxf %arg6, %210 : f32
            affine.yield %211 : f32
          }
          %209 = affine.for %arg5 = 0 to 128 iter_args(%arg6 = %cst_5) -> (f32) {
            %210 = affine.load %alloc_681[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.subf %210, %208 : f32
            %212 = math.exp %211 : f32
            %213 = arith.addf %arg6, %212 : f32
            affine.store %212, %alloc_682[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            affine.yield %213 : f32
          }
          affine.for %arg5 = 0 to 128 {
            %210 = affine.load %alloc_682[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
            %211 = arith.divf %210, %209 : f32
            affine.store %211, %alloc_682[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x128xf32>
          }
        }
      }
    }
    %alloc_683 = memref.alloc() {alignment = 16 : i64} : memref<1x12x128x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          affine.for %arg5 = 0 to 64 {
            %208 = affine.for %arg6 = 0 to 128 iter_args(%arg7 = %cst_5) -> (f32) {
              %209 = affine.load %alloc_682[%arg2, %arg3, %arg4, %arg6] : memref<1x12x128x128xf32>
              %210 = affine.load %alloc_678[%arg2, %arg3, %arg6, %arg5] : memref<1x12x128x64xf32>
              %211 = arith.mulf %209, %210 : f32
              %212 = arith.addf %arg7, %211 : f32
              affine.yield %212 : f32
            }
            affine.store %208, %alloc_683[%arg2, %arg3, %arg4, %arg5] : memref<1x12x128x64xf32>
          }
        }
      }
    }
    %alloc_684 = memref.alloc() {alignment = 16 : i64} : memref<1x128x12x64xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 12 {
        affine.for %arg4 = 0 to 128 {
          %208 = affine.apply #map11(%arg2, %arg3, %arg4)
          %209 = affine.apply #map12(%arg2, %arg3, %arg4)
          "krnl.memcpy"(%alloc_684, %alloc_683, %c64_i64, %209, %208) : (memref<1x128x12x64xf32>, memref<1x12x128x64xf32>, i64, index, index) -> ()
        }
      }
    }
    %reinterpret_cast_685 = memref.reinterpret_cast %alloc_684 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x12x64xf32> to memref<128x768xf32>
    %alloc_686 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_685[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %174[%arg4, %arg3] : memref<768x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %175[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_686[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_687 = memref.reinterpret_cast %alloc_686 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_688 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_666[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_687[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_688[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %206 = memref.get_global @constant_363 : memref<1xf32>
    %alloc_689 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_690 = arith.constant 1 : index
    %c128_691 = arith.constant 128 : index
    %c1_692 = arith.constant 1 : index
    %c0_693 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_689[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_688[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_689[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_689[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_689[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_689[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_694 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_689[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_689[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_694[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_695 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_688[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_688[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_695[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_696 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_697 = arith.constant 1 : index
    %c128_698 = arith.constant 128 : index
    %c1_699 = arith.constant 1 : index
    %c0_700 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_696[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_695[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_696[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_696[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_696[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_696[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_701 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_696[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_694[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %206[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_701[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_702 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_688[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_689[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_702[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_703 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_702[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_701[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %176[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %177[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_703[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %reinterpret_cast_704 = memref.reinterpret_cast %alloc_703 to offset: [0], sizes: [128, 768], strides: [768, 1] : memref<1x128x768xf32> to memref<128x768xf32>
    %alloc_705 = memref.alloc() {alignment = 128 : i64} : memref<128x3072xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 3072 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 768 {
          %211 = affine.load %reinterpret_cast_704[%arg2, %arg4] : memref<128x768xf32>
          %212 = affine.load %178[%arg4, %arg3] : memref<768x3072xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %179[%arg3] : memref<3072xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_705[%arg2, %arg3] : memref<128x3072xf32>
      }
    }
    %reinterpret_cast_706 = memref.reinterpret_cast %alloc_705 to offset: [0], sizes: [1, 128, 3072], strides: [393216, 3072, 1] : memref<128x3072xf32> to memref<1x128x3072xf32>
    %alloc_707 = memref.alloc() {alignment = 16 : i64} : memref<1x128x3072xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 3072 {
          %208 = affine.load %reinterpret_cast_706[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
          %209 = arith.mulf %208, %cst_3 : f32
          %210 = math.powf %208, %cst_2 : f32
          %211 = arith.mulf %210, %cst_1 : f32
          %212 = arith.addf %208, %211 : f32
          %213 = arith.mulf %212, %cst_0 : f32
          %214 = math.tanh %213 : f32
          %215 = arith.addf %214, %cst_6 : f32
          %216 = arith.mulf %209, %215 : f32
          affine.store %216, %alloc_707[%arg2, %arg3, %arg4] : memref<1x128x3072xf32>
        }
      }
    }
    %reinterpret_cast_708 = memref.reinterpret_cast %alloc_707 to offset: [0], sizes: [128, 3072], strides: [3072, 1] : memref<1x128x3072xf32> to memref<128x3072xf32>
    %alloc_709 = memref.alloc() {alignment = 128 : i64} : memref<128x768xf32>
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 768 {
        %alloca = memref.alloca() : memref<f32>
        affine.store %cst_5, %alloca[] : memref<f32>
        affine.for %arg4 = 0 to 3072 {
          %211 = affine.load %reinterpret_cast_708[%arg2, %arg4] : memref<128x3072xf32>
          %212 = affine.load %180[%arg4, %arg3] : memref<3072x768xf32>
          %213 = arith.mulf %211, %212 : f32
          %214 = affine.load %alloca[] : memref<f32>
          %215 = arith.addf %213, %214 : f32
          affine.store %215, %alloca[] : memref<f32>
        }
        %208 = affine.load %alloca[] : memref<f32>
        %209 = affine.load %181[%arg3] : memref<768xf32>
        %210 = arith.addf %208, %209 : f32
        affine.store %210, %alloc_709[%arg2, %arg3] : memref<128x768xf32>
      }
    }
    %reinterpret_cast_710 = memref.reinterpret_cast %alloc_709 to offset: [0], sizes: [1, 128, 768], strides: [98304, 768, 1] : memref<128x768xf32> to memref<1x128x768xf32>
    %alloc_711 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_703[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %reinterpret_cast_710[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.addf %208, %209 : f32
          affine.store %210, %alloc_711[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %207 = memref.get_global @constant_365 : memref<1xf32>
    %alloc_712 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_713 = arith.constant 1 : index
    %c128_714 = arith.constant 128 : index
    %c1_715 = arith.constant 1 : index
    %c0_716 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_712[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_711[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_712[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_712[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_712[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_712[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_717 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_712[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_712[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_717[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_718 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_711[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_711[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %210 = arith.mulf %208, %209 : f32
          affine.store %210, %alloc_718[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_719 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    %c1_720 = arith.constant 1 : index
    %c128_721 = arith.constant 128 : index
    %c1_722 = arith.constant 1 : index
    %c0_723 = arith.constant 0 : index
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          affine.store %cst_5, %alloc_719[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_718[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_719[%arg2, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.addf %209, %208 : f32
          affine.store %210, %alloc_719[%arg2, %arg3, %c0] : memref<1x128x1xf32>
        }
      }
    }
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_719[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
          %209 = arith.divf %208, %cst : f32
          affine.store %209, %alloc_719[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_724 = memref.alloc() {alignment = 16 : i64} : memref<1x128x1xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 1 {
          %208 = affine.load %alloc_719[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %209 = affine.load %alloc_717[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          %211 = affine.load %207[%c0] : memref<1xf32>
          %212 = arith.addf %210, %211 : f32
          %213 = math.sqrt %212 : f32
          %214 = arith.divf %cst_6, %213 : f32
          affine.store %214, %alloc_724[%arg2, %arg3, %arg4] : memref<1x128x1xf32>
        }
      }
    }
    %alloc_725 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_711[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_712[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.subf %208, %209 : f32
          affine.store %210, %alloc_725[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    %alloc_726 = memref.alloc() {alignment = 16 : i64} : memref<1x128x768xf32>
    affine.for %arg2 = 0 to 1 {
      affine.for %arg3 = 0 to 128 {
        affine.for %arg4 = 0 to 768 {
          %208 = affine.load %alloc_725[%c0, %arg3, %arg4] : memref<1x128x768xf32>
          %209 = affine.load %alloc_724[%c0, %arg3, %c0] : memref<1x128x1xf32>
          %210 = arith.mulf %208, %209 : f32
          %211 = affine.load %182[%arg4] : memref<768xf32>
          %212 = arith.mulf %210, %211 : f32
          %213 = affine.load %183[%arg4] : memref<768xf32>
          %214 = arith.addf %212, %213 : f32
          affine.store %214, %alloc_726[%arg2, %arg3, %arg4] : memref<1x128x768xf32>
        }
      }
    }
    return %alloc_726 : memref<1x128x768xf32>
  }

}

