_ZN5shelf6kernel12_GLOBAL__N_123gemm_exl3_w16a16_kernelILj1ELb0ELb1ELb1EEEvNS0_12Exl3GemmArgsEjPKt: ; @_ZN5shelf6kernel12_GLOBAL__N_123gemm_exl3_w16a16_kernelILj1ELb0ELb1ELb1EEEvNS0_12Exl3GemmArgsEjPKt
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x28
	s_load_b32 s6, s[0:1], 0x30
	v_lshrrev_b32_e32 v17, 5, v0
	s_lshl_b32 s12, ttmp7, 4
	v_and_b32_e32 v18, 15, v0
	v_and_b32_e32 v16, 31, v0
	v_bfe_u32 v19, v0, 4, 1
	v_lshlrev_b32_e32 v14, 1, v17
	s_mov_b32 s5, 0
	s_wait_kmcnt 0x0
	s_sub_co_i32 s2, s2, s12
	s_delay_alu instid0(SALU_CYCLE_1)
	s_min_u32 s13, s2, 16
	s_lshr_b32 s7, s6, 7
	s_cbranch_scc0 .LBB14_66
; %bb.1:
	s_clause 0x2
	s_load_b64 s[2:3], s[0:1], 0x70
	s_load_b32 s4, s[0:1], 0x2c
	s_load_b64 s[10:11], s[0:1], 0x58
	v_dual_mov_b32 v10, 0 :: v_dual_lshlrev_b32 v13, 1, v17
	v_lshlrev_b32_e32 v2, 4, v19
	v_lshl_or_b32 v1, v16, 3, 0x2800
	v_lshl_or_b32 v9, ttmp9, 3, v17
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_4) | instid1(VALU_DEP_4)
	v_or_b32_e32 v6, 1, v13
	v_mov_b32_e32 v11, v10
	v_and_b32_e32 v3, 0x7f, v0
	v_lshl_or_b32 v5, v18, 8, v2
	v_dual_mov_b32 v8, v10 :: v_dual_lshlrev_b32 v25, 1, v0
	v_mov_b32_e32 v12, v11
	v_dual_mov_b32 v11, v10 :: v_dual_lshlrev_b32 v4, 5, v18
	v_lshlrev_b32_e32 v3, 1, v3
	v_add_nc_u32_e32 v15, 0x2800, v5
	s_lshl_b32 s8, s13, 7
	s_wait_kmcnt 0x0
	v_add_co_u32 v4, s2, s2, v4
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v7, null, s3, 0, s2
	v_cmp_gt_u32_e64 s3, s13, v6
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v20, s2, v4, v2
	v_lshrrev_b32_e32 v2, 7, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v21, null, 0, v7, s2
	v_lshlrev_b32_e32 v4, 8, v6
	v_mov_b32_e32 v6, v10
	v_add_nc_u32_e32 v24, s12, v2
	v_mov_b32_e32 v2, v10
	v_add_co_u32 v22, s2, s10, v3
	v_lshlrev_b32_e32 v3, 9, v17
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v23, null, s11, 0, s2
	v_cmp_gt_u32_e64 s2, s13, v13
	v_add_nc_u32_e32 v27, v1, v4
	v_dual_mov_b32 v5, v10 :: v_dual_add_nc_u32 v26, v1, v3
	v_mov_b32_e32 v1, v10
	v_dual_mov_b32 v3, v10 :: v_dual_mov_b32 v4, v10
	v_mov_b32_e32 v7, v10
	s_lshr_b32 s9, s4, 4
	s_mov_b32 s10, 0
	v_cmp_gt_u32_e32 vcc_lo, s8, v0
	s_branch .LBB14_3
.LBB14_2:                               ;   in Loop: Header=BB14_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_lshl_b32 s4, s10, 3
	ds_load_b128 v[60:63], v15
	ds_load_b128 v[64:67], v15 offset:32
	ds_load_b128 v[68:71], v15 offset:64
	ds_load_b128 v[72:75], v15 offset:96
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[28:29], null, s4, s9, v[9:10]
	s_add_co_i32 s10, s10, 1
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s10, s7
	v_add_co_u32 v30, s4, v28, s9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, 0, v29, s4
	v_lshlrev_b64_e32 v[28:29], 9, v[28:29]
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_lshlrev_b64_e32 v[32:33], 9, v[30:31]
	v_add_co_u32 v30, s4, v30, s9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, 0, v31, s4
	v_add_co_u32 v28, s4, v20, v28
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, v21, v29, s4
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b64_e32 v[34:35], 9, v[30:31]
	v_add_co_u32 v30, s4, v30, s9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v31, null, 0, v31, s4
	v_add_co_u32 v32, s4, v20, v32
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v33, null, v21, v33, s4
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_lshlrev_b64_e32 v[36:37], 9, v[30:31]
	v_add_co_u32 v38, s4, v20, v34
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v39, null, v21, v35, s4
	v_add_co_u32 v40, s4, v20, v36
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v41, null, v21, v37, s4
	v_add_co_u32 v44, s4, v30, s9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, 0, v31, s4
	s_clause 0x3
	global_load_b128 v[28:31], v[28:29], off
	global_load_b128 v[32:35], v[32:33], off
	global_load_b128 v[36:39], v[38:39], off
	global_load_b128 v[40:43], v[40:41], off
	v_lshlrev_b64_e32 v[46:47], 9, v[44:45]
	v_add_co_u32 v44, s4, v44, s9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, 0, v45, s4
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_add_co_u32 v46, s4, v20, v46
	v_lshlrev_b64_e32 v[48:49], 9, v[44:45]
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v47, null, v21, v47, s4
	v_add_co_u32 v44, s4, v44, s9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v45, null, 0, v45, s4
	v_add_co_u32 v48, s4, v20, v48
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v49, null, v21, v49, s4
	v_add_co_u32 v52, s4, v44, s9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v53, null, 0, v45, s4
	v_lshlrev_b64_e32 v[54:55], 9, v[44:45]
	s_clause 0x1
	global_load_b128 v[44:47], v[46:47], off
	global_load_b128 v[48:51], v[48:49], off
	v_lshlrev_b64_e32 v[52:53], 9, v[52:53]
	v_add_co_u32 v54, s4, v20, v54
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v55, null, v21, v55, s4
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_u32 v56, s4, v20, v52
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v57, null, v21, v53, s4
	s_clause 0x1
	global_load_b128 v[52:55], v[54:55], off
	global_load_b128 v[56:59], v[56:57], off
	s_wait_loadcnt_dscnt 0x703
	v_wmma_f32_16x16x16_f16 v[1:8], v[28:31], v[60:63], v[1:8]
	ds_load_b128 v[28:31], v15 offset:128
	s_wait_loadcnt_dscnt 0x603
	v_wmma_f32_16x16x16_f16 v[1:8], v[32:35], v[64:67], v[1:8]
	ds_load_b128 v[32:35], v15 offset:160
	s_wait_loadcnt_dscnt 0x503
	v_wmma_f32_16x16x16_f16 v[1:8], v[36:39], v[68:71], v[1:8]
	ds_load_b128 v[36:39], v15 offset:192
	ds_load_b128 v[60:63], v15 offset:224
	s_wait_loadcnt_dscnt 0x404
	v_wmma_f32_16x16x16_f16 v[1:8], v[40:43], v[72:75], v[1:8]
	s_wait_loadcnt_dscnt 0x303
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[44:47], v[28:31], v[1:8]
	s_wait_loadcnt_dscnt 0x202
	v_wmma_f32_16x16x16_f16 v[1:8], v[48:51], v[32:35], v[1:8]
	s_wait_loadcnt_dscnt 0x101
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)
	v_wmma_f32_16x16x16_f16 v[1:8], v[52:55], v[36:39], v[1:8]
	s_wait_loadcnt_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[1:8], v[56:59], v[60:63], v[1:8]
	s_cbranch_scc1 .LBB14_10
.LBB14_3:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB14_5 Depth 2
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s11, vcc_lo
	s_cbranch_execz .LBB14_6
; %bb.4:                                ;   in Loop: Header=BB14_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s4, s10, 7
	v_dual_mov_b32 v30, v25 :: v_dual_mov_b32 v31, v24
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[14:15], s[4:5], 1
	v_mov_b32_e32 v32, v0
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v28, s4, v22, s14
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v29, null, s15, v23, s4
	s_mov_b32 s14, 0
.LBB14_5:                               ;   Parent Loop BB14_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mad_co_u64_u32 v[33:34], null, v31, s6, 0
	v_add_nc_u32_e32 v32, 0x100, v32
	v_add_nc_u32_e32 v31, 2, v31
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[33:34], 1, v[33:34]
	v_add_co_u32 v33, s4, v28, v33
	s_wait_alu depctr_va_sdst(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v34, null, v29, v34, s4
	v_cmp_le_u32_e64 s4, s8, v32
	global_load_d16_b16 v33, v[33:34], off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s14, s4, s14
	s_wait_loadcnt 0x0
	ds_store_b16 v30, v33 offset:10240
	v_add_nc_u32_e32 v30, 0x200, v30
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s14
	s_cbranch_execnz .LBB14_5
.LBB14_6:                               ;   in Loop: Header=BB14_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s11
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s4, s2
	s_cbranch_execz .LBB14_8
; %bb.7:                                ;   in Loop: Header=BB14_3 Depth=1
	ds_load_b64 v[28:29], v26
	s_wait_dscnt 0x0
	v_fma_mix_f32 v28, v28, 1.0, v28 op_sel:[0,0,1] op_sel_hi:[1,1,1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v28, v29, 1.0, v28 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v28, v29, 1.0, v28 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v28
.LBB14_8:                               ;   in Loop: Header=BB14_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	s_and_saveexec_b32 s4, s3
	s_cbranch_execz .LBB14_2
; %bb.9:                                ;   in Loop: Header=BB14_3 Depth=1
	ds_load_b64 v[28:29], v27
	s_wait_dscnt 0x0
	v_fma_mix_f32 v28, v28, 1.0, v28 op_sel:[0,0,1] op_sel_hi:[1,1,1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v28, v29, 1.0, v28 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v28, v29, 1.0, v28 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v12, v12, v28
	s_branch .LBB14_2
.LBB14_10:
	s_branch .LBB14_12
.LBB14_11:
	v_dual_mov_b32 v11, 0 :: v_dual_mov_b32 v12, 0
	v_mov_b32_e32 v13, v14
	s_delay_alu instid0(VALU_DEP_2)
	v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v4, v11
	v_dual_mov_b32 v2, v11 :: v_dual_mov_b32 v3, v11
	v_dual_mov_b32 v6, v11 :: v_dual_mov_b32 v5, v11
	v_dual_mov_b32 v8, v11 :: v_dual_mov_b32 v7, v11
.LBB14_12:
	ds_swizzle_b32 v10, v11 offset:swizzle(BITMASK_PERM,"1pppp")
	v_mbcnt_lo_u32_b32 v15, -1, 0
	s_clause 0x1
	s_load_b64 s[10:11], s[0:1], 0x10
	s_load_b64 s[8:9], s[0:1], 0x38
	v_cmp_eq_u32_e64 s2, 0, v16
	v_cmp_lt_u32_e32 vcc_lo, 23, v15
	v_cndmask_b32_e64 v9, 8, 0, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, 27, v15
	s_wait_dscnt 0x0
	v_add_f32_e32 v11, v11, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v10, 4, 0, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, 29, v15
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_nc_u32_e32 v10, v10, v15
	v_add_nc_u32_e32 v9, v9, v15
	v_lshlrev_b32_e32 v10, 2, v10
	s_delay_alu instid0(VALU_DEP_2)
	v_lshlrev_b32_e32 v9, 2, v9
	ds_bpermute_b32 v14, v9, v11
	s_wait_dscnt 0x0
	v_add_f32_e32 v14, v11, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v11, 2, 0, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 31, v15
	ds_bpermute_b32 v20, v10, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v21, null, 0, v15, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s13, v13
	s_and_b32 s4, s2, vcc_lo
	s_wait_dscnt 0x0
	v_dual_add_f32 v14, v14, v20 :: v_dual_add_nc_u32 v11, v11, v15
	s_delay_alu instid0(VALU_DEP_1)
	v_lshlrev_b32_e32 v11, 2, v11
	ds_bpermute_b32 v20, v11, v14
	s_wait_dscnt 0x0
	v_add_f32_e32 v20, v14, v20
	v_lshlrev_b32_e32 v14, 2, v21
	ds_bpermute_b32 v21, v14, v20
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB14_14
; %bb.13:
	s_wait_dscnt 0x0
	v_add_f32_e32 v20, v20, v21
	v_lshlrev_b32_e32 v21, 2, v13
	ds_store_b32 v21, v20 offset:14336
.LBB14_14:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	ds_swizzle_b32 v20, v12 offset:swizzle(BITMASK_PERM,"1pppp")
	s_load_b64 s[6:7], s[0:1], 0x20
	s_wait_dscnt 0x0
	v_add_f32_e32 v12, v12, v20
	ds_bpermute_b32 v9, v9, v12
	s_wait_dscnt 0x0
	v_add_f32_e32 v9, v12, v9
	ds_bpermute_b32 v10, v10, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v9, v9, v10
	ds_bpermute_b32 v10, v11, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v9, v9, v10
	ds_bpermute_b32 v10, v14, v9
	v_or_b32_e32 v14, 1, v13
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB14_17
; %bb.15:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s0, s13, v14
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB14_17
; %bb.16:
	s_wait_dscnt 0x0
	v_dual_add_f32 v9, v9, v10 :: v_dual_lshlrev_b32 v10, 2, v13
	ds_store_b32 v10, v9 offset:14340
.LBB14_17:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_dscnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v10, 0 :: v_dual_lshlrev_b32 v9, 9, v18
	v_lshlrev_b32_e32 v11, 6, v17
	v_lshlrev_b32_e32 v12, 5, v19
	s_lshl_b32 s0, ttmp9, 7
	v_and_b32_e32 v17, 8, v0
	v_and_b32_e32 v18, 1, v0
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s9, 0
	v_or3_b32 v11, v9, v11, v12
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_or_b32 v9, v16, 2, s0
	v_cmp_ne_u32_e64 s1, 0, v17
	v_cmp_eq_u32_e64 s4, 1, v18
	v_cmp_lt_u32_e64 s0, 15, v16
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_b128 v11, v[1:4]
	ds_store_b128 v11, v[5:8] offset:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_and_b32_e32 v2, 2, v0
	v_and_b32_e32 v3, 4, v0
	v_lshlrev_b64_e32 v[0:1], 1, v[9:10]
	v_lshlrev_b32_e32 v8, 4, v16
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cmp_ne_u32_e64 s3, 0, v2
	v_cmp_ne_u32_e64 s2, 0, v3
	s_delay_alu instid0(VALU_DEP_4)
	v_add_co_u32 v4, s5, s10, v0
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v5, null, s11, v1, s5
	s_cselect_b32 s5, -1, 0
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s9, vcc_lo
	s_cbranch_execz .LBB14_41
; %bb.18:
	v_lshlrev_b32_e32 v0, 2, v13
	v_lshl_or_b32 v1, v13, 9, v8
	v_xor_b32_e32 v7, 1, v15
	ds_load_b32 v6, v0 offset:14336
	ds_load_b128 v[0:3], v1
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	s_wait_dscnt 0x0
	v_dual_fmamk_f32 v0, v6, 0xc4bfc896, v0 :: v_dual_fmac_f32 v3, 0xc4bfc896, v6
	v_dual_fmamk_f32 v1, v6, 0xc4bfc896, v1 :: v_dual_fmamk_f32 v2, v6, 0xc4bfc896, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_cndmask_b32 v6, v15, v7 :: v_dual_sub_f32 v11, v0, v1
	v_dual_add_f32 v7, v0, v1 :: v_dual_add_f32 v12, v2, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v3, v2, v3 :: v_dual_lshlrev_b32 v6, 2, v6
	v_add_f32_e32 v0, v7, v12
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v1, v11, v3
	v_sub_f32_e32 v2, v7, v12
	v_sub_f32_e32 v3, v11, v3
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr12
                                        ; implicit-def: $vgpr11
	ds_bpermute_b32 v16, v6, v0
	ds_bpermute_b32 v17, v6, v1
	ds_bpermute_b32 v18, v6, v2
	ds_bpermute_b32 v19, v6, v3
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execz .LBB14_20
; %bb.19:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v11, v16, v0 :: v_dual_sub_f32 v12, v17, v1
	s_wait_dscnt 0x0
	v_dual_sub_f32 v6, v18, v2 :: v_dual_sub_f32 v7, v19, v3
                                        ; implicit-def: $vgpr0
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr2
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr3
                                        ; implicit-def: $vgpr19
.LBB14_20:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB14_22
; %bb.21:
	s_wait_dscnt 0x2
	v_dual_add_f32 v11, v0, v16 :: v_dual_add_f32 v12, v1, v17
	s_wait_dscnt 0x0
	v_dual_add_f32 v6, v2, v18 :: v_dual_add_f32 v7, v3, v19
.LBB14_22:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_xor_b32_e32 v0, 2, v15
                                        ; implicit-def: $vgpr2_vgpr3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v15, v0, vcc_lo
	v_lshlrev_b32_e32 v0, 2, v0
	s_wait_dscnt 0x1
	ds_bpermute_b32 v18, v0, v11
	s_wait_dscnt 0x1
	ds_bpermute_b32 v19, v0, v12
	ds_bpermute_b32 v16, v0, v6
	ds_bpermute_b32 v17, v0, v7
                                        ; implicit-def: $vgpr0_vgpr1
	s_and_saveexec_b32 s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execz .LBB14_24
; %bb.23:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v1, v19, v12 :: v_dual_sub_f32 v0, v18, v11
	s_wait_dscnt 0x0
	v_dual_sub_f32 v3, v17, v7 :: v_dual_sub_f32 v2, v16, v6
                                        ; implicit-def: $vgpr12
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr11
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr16
.LBB14_24:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB14_26
; %bb.25:
	s_wait_dscnt 0x2
	v_dual_add_f32 v1, v12, v19 :: v_dual_add_f32 v0, v11, v18
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v7, v17 :: v_dual_add_f32 v2, v6, v16
.LBB14_26:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_xor_b32_e32 v6, 4, v15
                                        ; implicit-def: $vgpr11_vgpr12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v15, v6, vcc_lo
	v_lshlrev_b32_e32 v6, 2, v6
	s_wait_dscnt 0x3
	ds_bpermute_b32 v18, v6, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v19, v6, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v6, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v17, v6, v3
                                        ; implicit-def: $vgpr6_vgpr7
	s_and_saveexec_b32 s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execz .LBB14_28
; %bb.27:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v7, v19, v1 :: v_dual_sub_f32 v6, v18, v0
	s_wait_dscnt 0x0
	v_dual_sub_f32 v12, v17, v3 :: v_dual_sub_f32 v11, v16, v2
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr16
.LBB14_28:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB14_30
; %bb.29:
	s_wait_dscnt 0x2
	v_dual_add_f32 v7, v1, v19 :: v_dual_add_f32 v6, v0, v18
	s_wait_dscnt 0x0
	v_dual_add_f32 v12, v3, v17 :: v_dual_add_f32 v11, v2, v16
.LBB14_30:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_xor_b32_e32 v0, 8, v15
                                        ; implicit-def: $vgpr2_vgpr3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v15, v0, vcc_lo
	v_lshlrev_b32_e32 v0, 2, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v18, v0, v6
	s_wait_dscnt 0x3
	ds_bpermute_b32 v19, v0, v7
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v0, v11
	s_wait_dscnt 0x3
	ds_bpermute_b32 v17, v0, v12
                                        ; implicit-def: $vgpr0_vgpr1
	s_and_saveexec_b32 s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execz .LBB14_32
; %bb.31:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v1, v19, v7 :: v_dual_sub_f32 v0, v18, v6
	s_wait_dscnt 0x0
	v_dual_sub_f32 v3, v17, v12 :: v_dual_sub_f32 v2, v16, v11
                                        ; implicit-def: $vgpr6_vgpr7
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr11_vgpr12
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr16
.LBB14_32:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB14_34
; %bb.33:
	s_wait_dscnt 0x2
	v_dual_add_f32 v1, v7, v19 :: v_dual_add_f32 v0, v6, v18
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v12, v17 :: v_dual_add_f32 v2, v11, v16
.LBB14_34:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_xor_b32_e32 v6, 16, v15
                                        ; implicit-def: $vgpr11_vgpr12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v15, v6, vcc_lo
	v_lshlrev_b32_e32 v6, 2, v6
	s_wait_dscnt 0x3
	ds_bpermute_b32 v18, v6, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v19, v6, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v6, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v17, v6, v3
                                        ; implicit-def: $vgpr6_vgpr7
	s_and_saveexec_b32 s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execz .LBB14_36
; %bb.35:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v7, v19, v1 :: v_dual_sub_f32 v6, v18, v0
	s_wait_dscnt 0x0
	v_dual_sub_f32 v12, v17, v3 :: v_dual_sub_f32 v11, v16, v2
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr16
.LBB14_36:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB14_38
; %bb.37:
	s_wait_dscnt 0x2
	v_dual_add_f32 v7, v1, v19 :: v_dual_add_f32 v6, v0, v18
	s_wait_dscnt 0x0
	v_dual_add_f32 v12, v3, v17 :: v_dual_add_f32 v11, v2, v16
.LBB14_38:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_wait_dscnt 0x0
	global_load_b64 v[16:17], v[4:5], off
	v_dual_mul_f32 v1, 0x3a1ccd09, v7 :: v_dual_add_nc_u32 v0, s12, v13
	v_dual_mul_f32 v2, 0x3a1ccd09, v6 :: v_dual_mul_f32 v3, 0x3a1ccd09, v12
	v_mul_f32_e32 v11, 0x3a1ccd09, v11
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[6:7], null, v0, s8, v[9:10]
	s_and_b32 vcc_lo, exec_lo, s5
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v1, v16, v1, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v0, v16, v2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v3, v17, v3, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v2, v17, v11, neg(0) op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB14_67
; %bb.39:
	v_lshlrev_b64_e32 v[11:12], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc_lo, s6, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s7, v12, vcc_lo
	global_store_b128 v[11:12], v[0:3], off
	s_cbranch_execnz .LBB14_41
.LBB14_40:
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]
	v_bfe_u32 v12, v0, 16, 1
	v_bfe_u32 v16, v2, 16, 1
	v_or_b32_e32 v18, 0x400000, v0
	v_bfe_u32 v11, v1, 16, 1
	v_or_b32_e32 v20, 0x400000, v2
	v_add_co_u32 v6, vcc_lo, s6, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s7, v7, vcc_lo
	v_add3_u32 v12, v12, v0, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v0, v0
	v_add3_u32 v16, v16, v2, 0x7fff
	v_add3_u32 v11, v11, v1, 0x7fff
	v_or_b32_e32 v17, 0x400000, v1
	v_bfe_u32 v19, v3, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, v12, v18, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v2, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_add3_u32 v18, v19, v3, 0x7fff
	v_or_b32_e32 v19, 0x400000, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v16, v20, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v1, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v11, v17, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v3, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v18, v19, vcc_lo
	v_mov_b16_e32 v1.l, v2.h
	v_mov_b16_e32 v0.l, v12.h
	global_store_b64 v[6:7], v[0:1], off
.LBB14_41:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s9, exec_lo
	v_cmpx_gt_u32_e64 s13, v14
	s_cbranch_execz .LBB14_65
; %bb.42:
	v_lshlrev_b32_e32 v0, 2, v13
	v_lshl_or_b32 v1, v14, 9, v8
	v_xor_b32_e32 v7, 1, v15
	ds_load_b32 v6, v0 offset:14340
	ds_load_b128 v[0:3], v1
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	s_wait_dscnt 0x0
	v_dual_fmamk_f32 v0, v6, 0xc4bfc896, v0 :: v_dual_fmac_f32 v3, 0xc4bfc896, v6
	v_dual_fmamk_f32 v1, v6, 0xc4bfc896, v1 :: v_dual_fmamk_f32 v2, v6, 0xc4bfc896, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_cndmask_b32 v6, v15, v7 :: v_dual_add_f32 v7, v0, v1
	v_dual_sub_f32 v8, v0, v1 :: v_dual_add_f32 v11, v2, v3
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_sub_f32 v3, v2, v3 :: v_dual_lshlrev_b32 v6, 2, v6
	v_add_f32_e32 v0, v7, v11
	s_delay_alu instid0(VALU_DEP_2)
	v_add_f32_e32 v1, v8, v3
	v_sub_f32_e32 v2, v7, v11
	v_sub_f32_e32 v3, v8, v3
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr11
                                        ; implicit-def: $vgpr8
	ds_bpermute_b32 v12, v6, v0
	ds_bpermute_b32 v13, v6, v1
	ds_bpermute_b32 v16, v6, v2
	ds_bpermute_b32 v17, v6, v3
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s9
	s_cbranch_execz .LBB14_44
; %bb.43:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v8, v12, v0 :: v_dual_sub_f32 v11, v13, v1
	s_wait_dscnt 0x0
	v_dual_sub_f32 v6, v16, v2 :: v_dual_sub_f32 v7, v17, v3
                                        ; implicit-def: $vgpr0
                                        ; implicit-def: $vgpr12
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr2
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr3
                                        ; implicit-def: $vgpr17
.LBB14_44:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
	s_cbranch_execz .LBB14_46
; %bb.45:
	s_wait_dscnt 0x2
	v_dual_add_f32 v8, v0, v12 :: v_dual_add_f32 v11, v1, v13
	s_wait_dscnt 0x0
	v_dual_add_f32 v6, v2, v16 :: v_dual_add_f32 v7, v3, v17
.LBB14_46:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_xor_b32_e32 v0, 2, v15
                                        ; implicit-def: $vgpr2_vgpr3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v15, v0, vcc_lo
	v_lshlrev_b32_e32 v0, 2, v0
	s_wait_dscnt 0x1
	ds_bpermute_b32 v16, v0, v8
	s_wait_dscnt 0x1
	ds_bpermute_b32 v17, v0, v11
	ds_bpermute_b32 v12, v0, v6
	ds_bpermute_b32 v13, v0, v7
                                        ; implicit-def: $vgpr0_vgpr1
	s_and_saveexec_b32 s4, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s4
	s_cbranch_execz .LBB14_48
; %bb.47:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v1, v17, v11 :: v_dual_sub_f32 v0, v16, v8
	s_wait_dscnt 0x0
	v_dual_sub_f32 v3, v13, v7 :: v_dual_sub_f32 v2, v12, v6
                                        ; implicit-def: $vgpr11
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr12
.LBB14_48:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB14_50
; %bb.49:
	s_wait_dscnt 0x2
	v_dual_add_f32 v1, v11, v17 :: v_dual_add_f32 v0, v8, v16
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v7, v13 :: v_dual_add_f32 v2, v6, v12
.LBB14_50:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s3
	v_xor_b32_e32 v6, 4, v15
                                        ; implicit-def: $vgpr11_vgpr12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v15, v6, vcc_lo
	v_lshlrev_b32_e32 v6, 2, v6
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v6, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v17, v6, v1
	ds_bpermute_b32 v8, v6, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v13, v6, v3
                                        ; implicit-def: $vgpr6_vgpr7
	s_and_saveexec_b32 s3, s2
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s2, exec_lo, s3
	s_cbranch_execz .LBB14_52
; %bb.51:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v7, v17, v1 :: v_dual_sub_f32 v6, v16, v0
	s_wait_dscnt 0x0
	v_dual_sub_f32 v12, v13, v3 :: v_dual_sub_f32 v11, v8, v2
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr8
.LBB14_52:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB14_54
; %bb.53:
	s_wait_dscnt 0x2
	v_dual_add_f32 v7, v1, v17 :: v_dual_add_f32 v6, v0, v16
	s_wait_dscnt 0x0
	v_dual_add_f32 v12, v3, v13 :: v_dual_add_f32 v11, v2, v8
.LBB14_54:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s2
	v_xor_b32_e32 v0, 8, v15
                                        ; implicit-def: $vgpr2_vgpr3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v15, v0, vcc_lo
	v_lshlrev_b32_e32 v0, 2, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v0, v6
	s_wait_dscnt 0x3
	ds_bpermute_b32 v17, v0, v7
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v0, v11
	s_wait_dscnt 0x3
	ds_bpermute_b32 v13, v0, v12
                                        ; implicit-def: $vgpr0_vgpr1
	s_and_saveexec_b32 s2, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s1, exec_lo, s2
	s_cbranch_execz .LBB14_56
; %bb.55:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v1, v17, v7 :: v_dual_sub_f32 v0, v16, v6
	s_wait_dscnt 0x0
	v_dual_sub_f32 v3, v13, v12 :: v_dual_sub_f32 v2, v8, v11
                                        ; implicit-def: $vgpr6_vgpr7
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr11_vgpr12
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr8
.LBB14_56:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB14_58
; %bb.57:
	s_wait_dscnt 0x2
	v_dual_add_f32 v1, v7, v17 :: v_dual_add_f32 v0, v6, v16
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v12, v13 :: v_dual_add_f32 v2, v11, v8
.LBB14_58:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_xor_b32_e32 v6, 16, v15
                                        ; implicit-def: $vgpr11_vgpr12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v15, v6, vcc_lo
	v_lshlrev_b32_e32 v6, 2, v6
	ds_bpermute_b32 v15, v6, v0
	s_wait_dscnt 0x4
	ds_bpermute_b32 v16, v6, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v6, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v13, v6, v3
                                        ; implicit-def: $vgpr6_vgpr7
	s_and_saveexec_b32 s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s1
	s_cbranch_execz .LBB14_60
; %bb.59:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v7, v16, v1 :: v_dual_sub_f32 v6, v15, v0
	s_wait_dscnt 0x0
	v_dual_sub_f32 v12, v13, v3 :: v_dual_sub_f32 v11, v8, v2
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr13
                                        ; implicit-def: $vgpr8
.LBB14_60:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execz .LBB14_62
; %bb.61:
	s_wait_dscnt 0x2
	v_dual_add_f32 v7, v1, v16 :: v_dual_add_f32 v6, v0, v15
	s_wait_dscnt 0x0
	v_dual_add_f32 v12, v3, v13 :: v_dual_add_f32 v11, v2, v8
.LBB14_62:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_dscnt 0x2
	global_load_b64 v[15:16], v[4:5], off
	v_dual_mul_f32 v1, 0x3a1ccd09, v7 :: v_dual_add_nc_u32 v0, s12, v14
	v_dual_mul_f32 v2, 0x3a1ccd09, v6 :: v_dual_mul_f32 v3, 0x3a1ccd09, v12
	v_mul_f32_e32 v6, 0x3a1ccd09, v11
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[4:5], null, v0, s8, v[9:10]
	s_and_not1_b32 vcc_lo, exec_lo, s5
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v1, v15, v1, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v0, v15, v2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v3, v16, v3, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v2, v16, v6, neg(0) op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB14_68
; %bb.63:
	v_lshlrev_b64_e32 v[6:7], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s6, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s7, v7, vcc_lo
	global_store_b128 v[6:7], v[0:3], off
	s_cbranch_execnz .LBB14_65
.LBB14_64:
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]
	v_bfe_u32 v7, v0, 16, 1
	s_wait_dscnt 0x1
	v_bfe_u32 v8, v2, 16, 1
	v_or_b32_e32 v10, 0x400000, v0
	v_bfe_u32 v6, v1, 16, 1
	v_or_b32_e32 v12, 0x400000, v2
	v_add_co_u32 v4, vcc_lo, s6, v4
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, s7, v5, vcc_lo
	v_add3_u32 v7, v7, v0, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v0, v0
	v_add3_u32 v8, v8, v2, 0x7fff
	v_add3_u32 v6, v6, v1, 0x7fff
	v_or_b32_e32 v9, 0x400000, v1
	v_bfe_u32 v11, v3, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v7, v7, v10, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v2, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_add3_u32 v10, v11, v3, 0x7fff
	v_or_b32_e32 v11, 0x400000, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v8, v12, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v1, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v6, v9, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v3, v3
	v_mov_b16_e32 v0.l, v7.h
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v10, v11, vcc_lo
	v_mov_b16_e32 v1.l, v2.h
	global_store_b64 v[4:5], v[0:1], off
.LBB14_65:
	s_endpgm
.LBB14_66:
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr12
                                        ; implicit-def: $vgpr13
	s_cbranch_execnz .LBB14_11
	s_branch .LBB14_12
.LBB14_67:
	s_branch .LBB14_40
.LBB14_68:
	s_branch .LBB14_64
.Lfunc_end14:
