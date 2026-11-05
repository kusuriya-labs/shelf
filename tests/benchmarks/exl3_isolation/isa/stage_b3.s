_ZN5shelf6kernel12_GLOBAL__N_123gemm_exl3_w16a16_kernelILj1ELb0ELb1ELb0EEEvNS0_12Exl3GemmArgsEjPKt: ; @_ZN5shelf6kernel12_GLOBAL__N_123gemm_exl3_w16a16_kernelILj1ELb0ELb1ELb0EEEvNS0_12Exl3GemmArgsEjPKt
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_clause 0x1
	s_load_b32 s2, s[0:1], 0x28
	s_load_b32 s62, s[0:1], 0x30
	v_lshrrev_b32_e32 v33, 5, v0
	s_lshl_b32 s60, ttmp7, 4
	v_and_b32_e32 v34, 15, v0
	v_and_b32_e32 v32, 31, v0
	v_bfe_u32 v35, v0, 4, 1
	v_lshlrev_b32_e32 v36, 1, v33
	s_mov_b32 s53, 0
	s_wait_kmcnt 0x0
	s_sub_co_i32 s2, s2, s60
	s_delay_alu instid0(SALU_CYCLE_1)
	s_min_u32 s61, s2, 16
	s_lshr_b32 s63, s62, 7
	s_cbranch_scc0 .LBB10_142
; %bb.1:
	s_clause 0x3
	s_load_b64 s[34:35], s[0:1], 0x40
	s_load_b64 s[56:57], s[0:1], 0x0
	s_load_b32 s3, s[0:1], 0x2c
	s_load_b64 s[50:51], s[0:1], 0x58
	v_dual_mov_b32 v10, 0 :: v_dual_lshlrev_b32 v31, 1, v33
	v_lshlrev_b32_e32 v2, 3, v32
	s_lshl_b32 s64, s61, 7
	v_lshlrev_b32_e32 v1, 2, v32
	v_lshlrev_b32_e32 v6, 10, v33
	v_lshl_or_b32 v9, ttmp9, 3, v33
	v_or_b32_e32 v3, 1, v2
	v_or_b32_e32 v5, 2, v2
	v_or_b32_e32 v29, 0x2800, v2
	v_mad_i32_i24 v8, 0xfffffd00, v33, v6
	s_mov_b32 s55, s53
	v_lshlrev_b32_e32 v59, 9, v33
	v_lshlrev_b32_e32 v43, 1, v0
	s_delay_alu instid0(VALU_DEP_3)
	v_or_b32_e32 v58, v8, v1
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s34, 4
	v_mul_lo_u32 v3, s34, v3
	s_cselect_b32 s2, -1, 0
	s_cmp_lg_u32 s35, 0
	v_add_co_u32 v50, s58, s56, v1
	s_cselect_b32 s4, -1, 0
	s_lshl_b32 s5, s34, 4
	s_or_b32 s65, s2, s4
	s_cmp_eq_u32 s35, 0
	v_add_co_ci_u32_e64 v51, null, s57, 0, s58
	s_cselect_b32 s35, -1, 0
	v_add_nc_u32_e32 v59, v29, v59
	s_wait_alu depctr_sa_sdst(0)
	v_cndmask_b32_e64 v4, v1, 0, s35
	s_and_b32 s2, s35, exec_lo
	s_cselect_b32 s4, 0, 8
	s_cselect_b32 s37, 0, 0x80
	s_lshl_b32 s2, s34, 8
	s_or_b32 s52, s4, s5
	s_or_b32 s37, s37, s2
	s_lshr_b32 s66, s52, 1
	v_add3_u32 v3, v4, v3, s37
	v_lshrrev_b32_e32 v4, 1, v5
	v_mul_lo_u32 v5, s34, v5
	s_lshr_b32 s54, s3, 4
	v_cmp_gt_u32_e64 s2, s64, v0
	v_add_nc_u32_e32 v7, -16, v3
	v_cndmask_b32_e64 v4, v4, 0, s35
	v_add_nc_u32_e32 v18, -1, v3
	s_lshl_b32 s58, s54, 1
	v_cmp_gt_u32_e64 s3, s66, v32
	v_lshrrev_b32_e32 v7, 5, v7
	v_add3_u32 v5, v4, v5, s37
	v_lshrrev_b32_e32 v20, 5, v18
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_add_nc_u32_e32 v11, 1, v7
	v_add_nc_u32_e32 v12, 2, v7
	v_add_nc_u32_e32 v13, 3, v7
	v_subrev_nc_u32_e32 v14, s66, v7
	v_cmp_le_u32_e32 vcc_lo, s66, v11
	s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_min_u32_e32 v14, v7, v14
	v_cndmask_b32_e64 v15, 0, s66, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s66, v12
	v_subrev_nc_u32_e32 v19, s66, v14
	s_delay_alu instid0(VALU_DEP_3)
	v_sub_nc_u32_e32 v11, v11, v15
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v16, 0, s66, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s66, v13
	v_sub_nc_u32_e32 v15, v7, v15
	v_min_u32_e32 v14, v14, v19
	s_delay_alu instid0(VALU_DEP_4)
	v_sub_nc_u32_e32 v12, v12, v16
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v17, 0, s66, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s66, v11
	v_sub_nc_u32_e32 v16, v7, v16
	v_lshl_add_u32 v37, v14, 2, v8
	v_add_nc_u32_e32 v14, -16, v5
	v_sub_nc_u32_e32 v13, v13, v17
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v11, 0, s66, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s66, v12
	v_sub_nc_u32_e32 v17, v7, v17
	s_delay_alu instid0(VALU_DEP_3)
	v_sub_nc_u32_e32 v11, v15, v11
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v12, 0, s66, vcc_lo
	v_and_b32_e32 v15, 0xffffffe0, v18
	v_cmp_le_u32_e32 vcc_lo, s66, v13
	v_lshl_add_u32 v38, v11, 2, v8
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_nc_u32_e32 v12, v16, v12
	v_sub_nc_u32_e32 v3, v15, v3
	v_or_b32_e32 v15, 3, v2
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v13, 0, s66, vcc_lo
	v_and_b32_e32 v11, 3, v0
	v_lshl_add_u32 v39, v12, 2, v8
	v_add_nc_u32_e32 v12, -1, v5
	v_add_nc_u32_e32 v41, 32, v3
	v_lshrrev_b32_e32 v3, 5, v14
	v_mul_lo_u32 v14, s34, v15
	v_sub_nc_u32_e32 v13, v17, v13
	v_and_b32_e32 v16, 0xffffffe0, v12
	v_lshrrev_b32_e32 v12, 5, v12
	v_sub_nc_u32_e32 v3, v3, v7
	v_lshlrev_b32_e32 v11, 2, v11
	v_lshl_add_u32 v40, v13, 2, v8
	v_sub_nc_u32_e32 v5, v16, v5
	v_sub_nc_u32_e32 v13, v20, v7
	v_cmp_eq_u32_e64 s7, 1, v3
	v_cmp_eq_u32_e64 s8, 2, v3
	v_cmp_eq_u32_e64 s9, 3, v3
	v_or_b32_e32 v3, 4, v2
	v_add3_u32 v4, v4, v14, s37
	v_add_nc_u32_e32 v42, 32, v5
	v_sub_nc_u32_e32 v5, v12, v7
	v_cmp_eq_u32_e64 s4, 1, v13
	v_lshrrev_b32_e32 v14, 1, v3
	v_cmp_eq_u32_e64 s5, 2, v13
	v_cmp_eq_u32_e64 s6, 3, v13
	v_add_nc_u32_e32 v13, -16, v4
	v_mul_lo_u32 v3, s34, v3
	v_cmp_eq_u32_e64 s10, 1, v5
	v_cmp_eq_u32_e64 s11, 2, v5
	v_cmp_eq_u32_e64 s12, 3, v5
	v_cndmask_b32_e64 v5, v14, 0, s35
	v_lshrrev_b32_e32 v13, 5, v13
	v_add_nc_u32_e32 v12, -1, v4
	v_mad_co_u64_u32 v[20:21], null, s54, 6, v[9:10]
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_add3_u32 v3, v5, v3, s37
	v_sub_nc_u32_e32 v13, v13, v7
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_and_b32_e32 v15, 0xffffffe0, v12
	v_lshrrev_b32_e32 v12, 5, v12
	v_add_nc_u32_e32 v5, -1, v3
	s_delay_alu instid0(VALU_DEP_4)
	v_cmp_eq_u32_e64 s13, 1, v13
	v_cmp_eq_u32_e64 s14, 2, v13
	v_cmp_eq_u32_e64 s15, 3, v13
	v_or_b32_e32 v13, 5, v2
	v_sub_nc_u32_e32 v4, v15, v4
	v_and_b32_e32 v14, 0xffffffe0, v5
	v_or_b32_e32 v15, 2, v1
	v_lshrrev_b32_e32 v5, 5, v5
	v_mul_lo_u32 v13, s34, v13
	v_add_nc_u32_e32 v44, 32, v4
	v_sub_nc_u32_e32 v4, v12, v7
	v_add_nc_u32_e32 v12, -16, v3
	v_sub_nc_u32_e32 v3, v14, v3
	v_cndmask_b32_e64 v14, v15, 0, s35
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)
	v_cmp_eq_u32_e64 s16, 1, v4
	v_cmp_eq_u32_e64 s17, 2, v4
	v_add_nc_u32_e32 v45, 32, v3
	s_delay_alu instid0(VALU_DEP_4)
	v_add3_u32 v3, v14, v13, s37
	v_cmp_eq_u32_e64 s18, 3, v4
	v_sub_nc_u32_e32 v4, v5, v7
	v_or_b32_e32 v5, 6, v2
	v_lshrrev_b32_e32 v12, 5, v12
	v_add_nc_u32_e32 v13, -1, v3
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_cmp_eq_u32_e64 s22, 1, v4
	v_lshrrev_b32_e32 v14, 1, v5
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sub_nc_u32_e32 v12, v12, v7
	v_and_b32_e32 v15, 0xffffffe0, v13
	v_mul_lo_u32 v5, s34, v5
	v_lshrrev_b32_e32 v13, 5, v13
	v_cndmask_b32_e64 v14, v14, 0, s35
	v_cmp_eq_u32_e64 s19, 1, v12
	v_cmp_eq_u32_e64 s20, 2, v12
	v_cmp_eq_u32_e64 s21, 3, v12
	v_add_nc_u32_e32 v12, -16, v3
	v_sub_nc_u32_e32 v3, v15, v3
	v_cmp_eq_u32_e64 s23, 2, v4
	v_cmp_eq_u32_e64 s24, 3, v4
	v_add3_u32 v4, v14, v5, s37
	v_lshrrev_b32_e32 v12, 5, v12
	v_add_nc_u32_e32 v46, 32, v3
	v_sub_nc_u32_e32 v3, v13, v7
	v_or_b32_e32 v13, 7, v2
	v_add_nc_u32_e32 v5, -1, v4
	v_or_b32_e32 v15, 3, v1
	v_sub_nc_u32_e32 v12, v12, v7
	v_cmp_eq_u32_e64 s28, 1, v3
	v_mul_lo_u32 v13, s34, v13
	v_and_b32_e32 v14, 0xffffffe0, v5
	v_cmp_eq_u32_e64 s29, 2, v3
	v_cmp_eq_u32_e64 s30, 3, v3
	v_cndmask_b32_e64 v3, v15, 0, s35
	v_cmp_eq_u32_e64 s25, 1, v12
	v_cmp_eq_u32_e64 s26, 2, v12
	v_cmp_eq_u32_e64 s27, 3, v12
	v_add_nc_u32_e32 v12, -16, v4
	v_sub_nc_u32_e32 v4, v14, v4
	v_lshrrev_b32_e32 v5, 5, v5
	v_add_nc_u32_e32 v2, 8, v2
	v_add3_u32 v3, v3, v13, s37
	v_lshrrev_b32_e32 v12, 5, v12
	v_add_nc_u32_e32 v47, 32, v4
	v_sub_nc_u32_e32 v4, v5, v7
	v_lshrrev_b32_e32 v5, 1, v2
	v_add_nc_u32_e32 v13, -1, v3
	v_sub_nc_u32_e32 v12, v12, v7
	v_mul_lo_u32 v2, s34, v2
	v_cmp_eq_u32_e64 s36, 2, v4
	v_cndmask_b32_e64 v5, v5, 0, s35
	v_and_b32_e32 v14, 0xffffffe0, v13
	v_cmp_eq_u32_e64 s31, 1, v12
	v_cmp_eq_u32_e64 s33, 2, v12
	v_cmp_eq_u32_e64 s34, 3, v12
	v_add_nc_u32_e32 v12, -16, v3
	v_add3_u32 v2, v5, v2, s37
	v_sub_nc_u32_e32 v3, v14, v3
	v_cmp_eq_u32_e64 s35, 1, v4
	v_cmp_eq_u32_e64 s37, 3, v4
	v_lshrrev_b32_e32 v12, 5, v12
	v_lshrrev_b32_e32 v4, 5, v13
	v_add_nc_u32_e32 v48, 32, v3
	v_add_nc_u32_e32 v3, -16, v2
	v_lshrrev_b32_e32 v15, 7, v0
	v_sub_nc_u32_e32 v5, v12, v7
	v_add_nc_u32_e32 v12, -1, v2
	v_sub_nc_u32_e32 v4, v4, v7
	v_lshrrev_b32_e32 v3, 5, v3
	v_add_nc_u32_e32 v57, s60, v15
	v_cmp_eq_u32_e64 s38, 1, v5
	v_and_b32_e32 v13, 0xffffffe0, v12
	v_cmp_eq_u32_e64 s39, 2, v5
	v_sub_nc_u32_e32 v3, v3, v7
	v_cmp_eq_u32_e64 s40, 3, v5
	v_lshrrev_b32_e32 v5, 5, v12
	v_sub_nc_u32_e32 v2, v13, v2
	v_cmp_eq_u32_e64 s41, 1, v4
	v_cmp_eq_u32_e64 s44, 1, v3
	v_cmp_eq_u32_e64 s45, 2, v3
	v_cmp_eq_u32_e64 s46, 3, v3
	v_add_nc_u32_e32 v3, -1, v0
	v_add_nc_u32_e32 v49, 32, v2
	v_sub_nc_u32_e32 v2, v5, v7
	v_lshlrev_b32_e32 v7, 4, v35
	v_cmp_eq_u32_e64 s42, 2, v4
	v_and_b32_e32 v30, 31, v3
	v_and_b32_e32 v3, 0x7f, v0
	v_cmp_eq_u32_e64 s47, 1, v2
	v_cmp_eq_u32_e64 s48, 2, v2
	v_cmp_eq_u32_e64 s49, 3, v2
	v_lshl_or_b32 v2, v34, 8, v7
	v_lshlrev_b32_e32 v3, 1, v3
	v_cmp_eq_u32_e64 s43, 3, v4
	v_mad_co_u64_u32 v[4:5], null, s52, v9, 0
	v_lshlrev_b32_e32 v12, 5, v34
	v_add_nc_u32_e32 v52, 0x2800, v2
	v_lshlrev_b32_e32 v2, 3, v0
	v_add_co_u32 v53, s50, s50, v3
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v54, null, s51, 0, s50
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_3)
	v_and_b32_e32 v14, 0xe0, v2
	v_add_co_u32 v2, s51, s54, v9
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v16, null, 0, 0, s51
	v_or3_b32 v55, v6, v14, v11
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[2:3], null, v2, s52, 0
	v_or3_b32 v56, v6, v12, v7
	v_lshlrev_b64_e32 v[5:6], 1, v[4:5]
	v_or_b32_e32 v13, 1, v31
	v_mad_co_u64_u32 v[7:8], null, s54, 3, v[9:10]
	v_cmp_gt_u32_e64 s50, s61, v31
	v_lshlrev_b32_e32 v61, 2, v30
	v_mad_co_u64_u32 v[3:4], null, v16, s52, v[3:4]
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v4, s58, s58, v9
	v_cmp_gt_u32_e64 s51, s61, v13
	v_lshlrev_b32_e32 v60, 8, v13
	v_add_co_u32 v13, vcc_lo, v5, v1
	s_delay_alu instid0(VALU_DEP_4)
	v_mad_co_u64_u32 v[4:5], null, v4, s52, 0
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v6, null, 0, v6, vcc_lo
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v15, null, 0, 0, s58
	v_lshlrev_b64_e32 v[11:12], 1, v[2:3]
	v_add_co_u32 v13, vcc_lo, s56, v13
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s57, v6, vcc_lo
	v_mad_co_u64_u32 v[5:6], null, v15, s52, v[5:6]
	v_mad_co_u64_u32 v[2:3], null, v7, s52, 0
	s_mul_u64 s[58:59], s[54:55], s[52:53]
	s_lshl_b32 s55, s54, 2
	v_add_co_u32 v15, vcc_lo, v11, v1
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v11, s55, s55, v9
	v_lshlrev_b64_e32 v[6:7], 1, v[4:5]
	v_mad_co_u64_u32 v[3:4], null, v8, s52, v[3:4]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, 0, v12, vcc_lo
	v_mad_co_u64_u32 v[4:5], null, v11, s52, 0
	v_mad_co_u64_u32 v[11:12], null, s54, 5, v[9:10]
	s_wait_alu depctr_va_sdst(0)
	v_add_co_ci_u32_e64 v17, null, 0, 0, s55
	v_add_co_u32 v15, vcc_lo, s56, v15
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s57, v16, vcc_lo
	v_add_co_u32 v18, vcc_lo, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v19, null, 0, v7, vcc_lo
	v_lshlrev_b64_e32 v[7:8], 1, v[2:3]
	v_mad_co_u64_u32 v[5:6], null, v17, s52, v[5:6]
	v_mad_co_u64_u32 v[2:3], null, v11, s52, 0
	v_add_co_u32 v17, vcc_lo, s56, v18
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s57, v19, vcc_lo
	v_add_co_u32 v19, vcc_lo, v7, v1
	v_lshlrev_b64_e32 v[6:7], 1, v[4:5]
	v_mad_co_u64_u32 v[3:4], null, v12, s52, v[3:4]
	v_mad_co_u64_u32 v[4:5], null, v20, s52, 0
	v_mad_co_u64_u32 v[11:12], null, s54, 7, v[9:10]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	v_add_co_u32 v19, vcc_lo, s56, v19
	v_add_nc_u32_e32 v60, v29, v60
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_3)
	v_add_co_ci_u32_e64 v20, null, s57, v8, vcc_lo
	v_add_co_u32 v22, vcc_lo, v6, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v23, null, 0, v7, vcc_lo
	v_lshlrev_b64_e32 v[7:8], 1, v[2:3]
	v_mad_co_u64_u32 v[5:6], null, v21, s52, v[5:6]
	v_mad_co_u64_u32 v[2:3], null, v11, s52, 0
	v_mov_b32_e32 v11, v10
	v_add_co_u32 v21, vcc_lo, s56, v22
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, s57, v23, vcc_lo
	s_mov_b32 s55, 0
	v_lshlrev_b64_e32 v[5:6], 1, v[4:5]
	v_mad_co_u64_u32 v[3:4], null, v12, s52, v[3:4]
	v_mov_b32_e32 v12, v11
	v_add_co_u32 v7, vcc_lo, v7, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v8, null, 0, v8, vcc_lo
	v_mov_b32_e32 v11, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v23, vcc_lo, s56, v7
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, s57, v8, vcc_lo
	v_add_co_u32 v4, vcc_lo, v5, v1
	v_lshlrev_b64_e32 v[2:3], 1, v[2:3]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v5, null, 0, v6, vcc_lo
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_3)
	v_add_co_u32 v25, vcc_lo, s56, v4
	v_mov_b32_e32 v4, v10
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s57, v5, vcc_lo
	v_add_co_u32 v1, vcc_lo, v2, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v2, null, 0, v3, vcc_lo
	v_mov_b32_e32 v3, v10
	s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_3)
	v_add_co_u32 v27, vcc_lo, s56, v1
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s57, v2, vcc_lo
	v_dual_mov_b32 v1, v10 :: v_dual_mov_b32 v2, v10
	v_dual_mov_b32 v5, v10 :: v_dual_mov_b32 v6, v10
	v_dual_mov_b32 v7, v10 :: v_dual_mov_b32 v8, v10
	s_lshl_b64 s[56:57], s[58:59], 4
	s_branch .LBB10_3
.LBB10_2:                               ;   in Loop: Header=BB10_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s59
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB10_83
.LBB10_3:                               ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB10_5 Depth 2
                                        ;     Child Loop BB10_11 Depth 2
                                        ;     Child Loop BB10_21 Depth 2
                                        ;     Child Loop BB10_29 Depth 2
                                        ;     Child Loop BB10_38 Depth 2
                                        ;     Child Loop BB10_47 Depth 2
                                        ;     Child Loop BB10_56 Depth 2
                                        ;     Child Loop BB10_66 Depth 2
                                        ;     Child Loop BB10_76 Depth 2
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s58, s2
	s_cbranch_execz .LBB10_6
; %bb.4:                                ;   in Loop: Header=BB10_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b32 s52, s55, 7
	v_dual_mov_b32 v62, v43 :: v_dual_mov_b32 v63, v57
	s_wait_alu depctr_sa_sdst(0)
	s_lshl_b64 s[68:69], s[52:53], 1
	v_mov_b32_e32 v64, v0
	s_wait_alu depctr_sa_sdst(0)
	v_add_co_u32 v29, vcc_lo, v53, s68
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, s69, v54, vcc_lo
	s_mov_b32 s52, 0
.LBB10_5:                               ;   Parent Loop BB10_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	v_mad_co_u64_u32 v[65:66], null, v63, s62, 0
	v_add_nc_u32_e32 v64, 0x100, v64
	v_add_nc_u32_e32 v63, 2, v63
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[65:66], 1, v[65:66]
	v_add_co_u32 v65, vcc_lo, v29, v65
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v66, null, v30, v66, vcc_lo
	v_cmp_le_u32_e32 vcc_lo, s64, v64
	global_load_d16_b16 v65, v[65:66], off
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s52, vcc_lo, s52
	s_wait_loadcnt 0x0
	ds_store_b16 v62, v65 offset:10240
	v_add_nc_u32_e32 v62, 0x200, v62
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s52
	s_cbranch_execnz .LBB10_5
.LBB10_6:                               ;   in Loop: Header=BB10_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s58
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	s_and_saveexec_b32 s52, s50
	s_cbranch_execnz .LBB10_13
; %bb.7:                                ;   in Loop: Header=BB10_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s52
	s_and_saveexec_b32 s52, s51
	s_cbranch_execnz .LBB10_14
.LBB10_8:                               ;   in Loop: Header=BB10_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s52
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s65
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_15
.LBB10_9:                               ;   in Loop: Header=BB10_3 Depth=1
	s_and_saveexec_b32 s58, s3
	s_cbranch_execz .LBB10_12
; %bb.10:                               ;   in Loop: Header=BB10_3 Depth=1
	v_dual_mov_b32 v30, v14 :: v_dual_mov_b32 v29, v13
	v_dual_mov_b32 v62, v58 :: v_dual_mov_b32 v63, v32
	s_mov_b32 s59, 0
.LBB10_11:                              ;   Parent Loop BB10_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_b32 v64, v[29:30], off
	v_add_nc_u32_e32 v63, 32, v63
	v_add_co_u32 v29, vcc_lo, 0x80, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, 0, v30, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s52, s66, v63
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s59, s52, s59
	s_wait_loadcnt 0x0
	ds_store_b32 v62, v64 offset:8192
	v_add_nc_u32_e32 v62, 0x80, v62
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s59
	s_cbranch_execnz .LBB10_11
.LBB10_12:                              ;   in Loop: Header=BB10_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s58
	; wave barrier
	ds_load_b32 v30, v37 offset:8192
	ds_load_b32 v71, v38 offset:8196
	ds_load_b32 v72, v39 offset:8200
	ds_load_b32 v73, v40 offset:8204
	s_mov_b32 s59, -1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v29, v30, v71, s4
	v_cndmask_b32_e64 v62, v30, v71, s7
	v_cndmask_b32_e64 v63, v30, v71, s10
	v_cndmask_b32_e64 v64, v30, v71, s13
	v_cndmask_b32_e64 v65, v30, v71, s16
	v_cndmask_b32_e64 v66, v30, v71, s19
	v_cndmask_b32_e64 v67, v30, v71, s22
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v29, v29, v72, s5
	v_cndmask_b32_e64 v68, v30, v71, s25
	v_cndmask_b32_e64 v69, v30, v71, s28
	v_cndmask_b32_e64 v62, v62, v72, s8
	v_cndmask_b32_e64 v77, v63, v72, s11
	v_cndmask_b32_e64 v64, v64, v72, s14
	v_cndmask_b32_e64 v78, v65, v72, s17
	v_cndmask_b32_e64 v66, v66, v72, s20
	v_cndmask_b32_e64 v79, v67, v72, s23
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v29, v29, v73, s6
	v_cndmask_b32_e64 v70, v30, v71, s31
	v_cndmask_b32_e64 v80, v68, v72, s26
	v_cndmask_b32_e64 v81, v69, v72, s29
	v_cndmask_b32_e64 v63, v62, v73, s9
	v_cndmask_b32_e64 v62, v77, v73, s12
	v_cndmask_b32_e64 v65, v64, v73, s15
	v_cndmask_b32_e64 v64, v78, v73, s18
	v_cndmask_b32_e64 v67, v66, v73, s21
	v_lshrrev_b64 v[68:69], v41, v[29:30]
	v_cndmask_b32_e64 v66, v79, v73, s24
	v_cndmask_b32_e64 v82, v70, v72, s33
	v_cndmask_b32_e64 v70, v80, v73, s27
	v_cndmask_b32_e64 v69, v81, v73, s30
	v_lshrrev_b64 v[62:63], v42, v[62:63]
	v_cndmask_b32_e64 v75, v30, v71, s38
	v_cndmask_b32_e64 v76, v30, v71, s41
	v_lshrrev_b64 v[63:64], v44, v[64:65]
	v_lshrrev_b64 v[64:65], v45, v[66:67]
	v_cndmask_b32_e64 v74, v30, v71, s35
	v_and_b32_e32 v68, 0xffff, v68
	v_lshrrev_b64 v[65:66], v46, v[69:70]
	v_cndmask_b32_e64 v29, v75, v72, s39
	v_and_b32_e32 v70, 0xffff, v62
	v_cndmask_b32_e64 v62, v30, v71, s44
	v_cndmask_b32_e64 v66, v30, v71, s47
	v_cndmask_b32_e64 v67, v76, v72, s42
	v_mul_lo_u32 v75, 0x83dcd12d, v68
	v_cndmask_b32_e64 v68, v74, v72, s36
	v_cndmask_b32_e64 v30, v29, v73, s40
	v_cndmask_b32_e64 v62, v62, v72, s45
	v_cndmask_b32_e64 v71, v66, v72, s48
	v_cndmask_b32_e64 v29, v67, v73, s43
	v_cndmask_b32_e64 v67, v82, v73, s34
	v_cndmask_b32_e64 v66, v68, v73, s37
	v_cndmask_b32_e64 v69, v62, v73, s46
	v_cndmask_b32_e64 v68, v71, v73, s49
	v_lshrrev_b64 v[29:30], v48, v[29:30]
	v_and_b32_e32 v30, 0xffff, v63
	v_lshrrev_b64 v[62:63], v47, v[66:67]
	v_and_b32_e32 v63, 0xffff, v65
	v_lshrrev_b64 v[65:66], v49, v[68:69]
	v_and_b32_e32 v64, 0xffff, v64
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_and_b32_e32 v62, 0xffff, v62
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	v_and_b32_e32 v65, 0xffff, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_mul_lo_u32 v66, 0x83dcd12d, v70
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_sad_u8 v67, v75, 0, 0x64006400
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_u8 v29, v29, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v66, v66, 0, v67
	v_sad_hi_u8 v30, v64, 0, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v62, v62, 0, v63
	v_sad_hi_u8 v29, v65, 0, v29
	ds_store_2addr_b32 v55, v66, v30 offset1:4
	ds_store_2addr_b32 v55, v62, v29 offset0:64 offset1:68
	; wave barrier
	s_lshl_b32 s58, s55, 3
	s_branch .LBB10_17
.LBB10_13:                              ;   in Loop: Header=BB10_3 Depth=1
	ds_load_b64 v[29:30], v59
	s_wait_dscnt 0x0
	v_fma_mix_f32 v29, v29, 1.0, v29 op_sel:[0,0,1] op_sel_hi:[1,1,1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v29, v30, 1.0, v29 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v29, v30, 1.0, v29 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1)
	v_add_f32_e32 v11, v11, v29
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s52
	s_and_saveexec_b32 s52, s51
	s_cbranch_execz .LBB10_8
.LBB10_14:                              ;   in Loop: Header=BB10_3 Depth=1
	ds_load_b64 v[29:30], v60
	s_wait_dscnt 0x0
	v_fma_mix_f32 v29, v29, 1.0, v29 op_sel:[0,0,1] op_sel_hi:[1,1,1]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_fma_mix_f32 v29, v30, 1.0, v29 op_sel_hi:[1,1,0]
	v_fma_mix_f32 v29, v30, 1.0, v29 op_sel:[1,0,0] op_sel_hi:[1,1,0]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
	v_add_f32_e32 v12, v12, v29
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s52
	s_and_b32 vcc_lo, exec_lo, s65
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB10_9
.LBB10_15:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s59, 0
	s_lshl_b32 s58, s55, 3
	s_cbranch_execz .LBB10_17
; %bb.16:                               ;   in Loop: Header=BB10_3 Depth=1
	s_cmp_eq_u32 exec_lo, -1
	s_cbranch_scc1 .LBB10_31
.LBB10_17:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s52, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s59
	s_mov_b32 s59, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_2
.LBB10_18:                              ;   in Loop: Header=BB10_3 Depth=1
	ds_load_b128 v[62:65], v56
	ds_load_b128 v[66:69], v52
	s_and_b32 vcc_lo, exec_lo, s65
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[1:8], v[62:65], v[66:69], v[1:8]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_23
; %bb.19:                               ;   in Loop: Header=BB10_3 Depth=1
	s_and_saveexec_b32 s59, s3
	s_cbranch_execz .LBB10_22
; %bb.20:                               ;   in Loop: Header=BB10_3 Depth=1
	v_dual_mov_b32 v30, v16 :: v_dual_mov_b32 v29, v15
	v_dual_mov_b32 v62, v58 :: v_dual_mov_b32 v63, v32
	s_mov_b32 s67, 0
.LBB10_21:                              ;   Parent Loop BB10_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_b32 v64, v[29:30], off
	v_add_nc_u32_e32 v63, 32, v63
	v_add_co_u32 v29, vcc_lo, 0x80, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, 0, v30, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s52, s66, v63
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s67, s52, s67
	s_wait_loadcnt 0x0
	ds_store_b32 v62, v64 offset:8192
	v_add_nc_u32_e32 v62, 0x80, v62
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s67
	s_cbranch_execnz .LBB10_21
.LBB10_22:                              ;   in Loop: Header=BB10_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s59
	; wave barrier
	ds_load_b32 v30, v37 offset:8192
	ds_load_b32 v71, v38 offset:8196
	ds_load_b32 v72, v39 offset:8200
	ds_load_b32 v73, v40 offset:8204
	s_mov_b32 s59, -1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v29, v30, v71, s4
	v_cndmask_b32_e64 v62, v30, v71, s7
	v_cndmask_b32_e64 v63, v30, v71, s10
	v_cndmask_b32_e64 v64, v30, v71, s13
	v_cndmask_b32_e64 v65, v30, v71, s16
	v_cndmask_b32_e64 v66, v30, v71, s19
	v_cndmask_b32_e64 v67, v30, v71, s22
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v29, v29, v72, s5
	v_cndmask_b32_e64 v68, v30, v71, s25
	v_cndmask_b32_e64 v69, v30, v71, s28
	v_cndmask_b32_e64 v62, v62, v72, s8
	v_cndmask_b32_e64 v77, v63, v72, s11
	v_cndmask_b32_e64 v64, v64, v72, s14
	v_cndmask_b32_e64 v78, v65, v72, s17
	v_cndmask_b32_e64 v66, v66, v72, s20
	v_cndmask_b32_e64 v79, v67, v72, s23
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v29, v29, v73, s6
	v_cndmask_b32_e64 v70, v30, v71, s31
	v_cndmask_b32_e64 v80, v68, v72, s26
	v_cndmask_b32_e64 v81, v69, v72, s29
	v_cndmask_b32_e64 v63, v62, v73, s9
	v_cndmask_b32_e64 v62, v77, v73, s12
	v_cndmask_b32_e64 v65, v64, v73, s15
	v_cndmask_b32_e64 v64, v78, v73, s18
	v_cndmask_b32_e64 v67, v66, v73, s21
	v_lshrrev_b64 v[68:69], v41, v[29:30]
	v_cndmask_b32_e64 v66, v79, v73, s24
	v_cndmask_b32_e64 v82, v70, v72, s33
	v_cndmask_b32_e64 v70, v80, v73, s27
	v_cndmask_b32_e64 v69, v81, v73, s30
	v_lshrrev_b64 v[62:63], v42, v[62:63]
	v_cndmask_b32_e64 v75, v30, v71, s38
	v_cndmask_b32_e64 v76, v30, v71, s41
	v_lshrrev_b64 v[63:64], v44, v[64:65]
	v_lshrrev_b64 v[64:65], v45, v[66:67]
	v_cndmask_b32_e64 v74, v30, v71, s35
	v_and_b32_e32 v68, 0xffff, v68
	v_lshrrev_b64 v[65:66], v46, v[69:70]
	v_cndmask_b32_e64 v29, v75, v72, s39
	v_and_b32_e32 v70, 0xffff, v62
	v_cndmask_b32_e64 v62, v30, v71, s44
	v_cndmask_b32_e64 v66, v30, v71, s47
	v_cndmask_b32_e64 v67, v76, v72, s42
	v_mul_lo_u32 v75, 0x83dcd12d, v68
	v_cndmask_b32_e64 v68, v74, v72, s36
	v_cndmask_b32_e64 v30, v29, v73, s40
	v_cndmask_b32_e64 v62, v62, v72, s45
	v_cndmask_b32_e64 v71, v66, v72, s48
	v_cndmask_b32_e64 v29, v67, v73, s43
	v_cndmask_b32_e64 v67, v82, v73, s34
	v_cndmask_b32_e64 v66, v68, v73, s37
	v_cndmask_b32_e64 v69, v62, v73, s46
	v_cndmask_b32_e64 v68, v71, v73, s49
	v_lshrrev_b64 v[29:30], v48, v[29:30]
	v_and_b32_e32 v30, 0xffff, v63
	v_lshrrev_b64 v[62:63], v47, v[66:67]
	v_and_b32_e32 v63, 0xffff, v65
	v_lshrrev_b64 v[65:66], v49, v[68:69]
	v_and_b32_e32 v64, 0xffff, v64
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_and_b32_e32 v62, 0xffff, v62
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	v_and_b32_e32 v65, 0xffff, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_mul_lo_u32 v66, 0x83dcd12d, v70
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_sad_u8 v67, v75, 0, 0x64006400
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_u8 v29, v29, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v66, v66, 0, v67
	v_sad_hi_u8 v30, v64, 0, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v62, v62, 0, v63
	v_sad_hi_u8 v29, v65, 0, v29
	ds_store_2addr_b32 v55, v66, v30 offset0:128 offset1:132
	ds_store_2addr_b32 v55, v62, v29 offset0:192 offset1:196
	; wave barrier
	s_branch .LBB10_25
.LBB10_23:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s59, 0
	s_and_b32 vcc_lo, exec_lo, s52
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_25
; %bb.24:                               ;   in Loop: Header=BB10_3 Depth=1
	s_cmp_eq_u32 exec_lo, -1
	s_cbranch_scc1 .LBB10_40
.LBB10_25:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s52, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s59
	s_mov_b32 s59, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_2
.LBB10_26:                              ;   in Loop: Header=BB10_3 Depth=1
	ds_load_b128 v[62:65], v56 offset:512
	ds_load_b128 v[66:69], v52 offset:32
	s_and_b32 vcc_lo, exec_lo, s65
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[1:8], v[62:65], v[66:69], v[1:8]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_32
; %bb.27:                               ;   in Loop: Header=BB10_3 Depth=1
	s_and_saveexec_b32 s59, s3
	s_cbranch_execz .LBB10_30
; %bb.28:                               ;   in Loop: Header=BB10_3 Depth=1
	v_dual_mov_b32 v30, v18 :: v_dual_mov_b32 v29, v17
	v_dual_mov_b32 v62, v58 :: v_dual_mov_b32 v63, v32
	s_mov_b32 s67, 0
.LBB10_29:                              ;   Parent Loop BB10_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_b32 v64, v[29:30], off
	v_add_nc_u32_e32 v63, 32, v63
	v_add_co_u32 v29, vcc_lo, 0x80, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, 0, v30, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s52, s66, v63
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s67, s52, s67
	s_wait_loadcnt 0x0
	ds_store_b32 v62, v64 offset:8192
	v_add_nc_u32_e32 v62, 0x80, v62
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s67
	s_cbranch_execnz .LBB10_29
.LBB10_30:                              ;   in Loop: Header=BB10_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s59
	; wave barrier
	ds_load_b32 v30, v37 offset:8192
	ds_load_b32 v71, v38 offset:8196
	ds_load_b32 v72, v39 offset:8200
	ds_load_b32 v73, v40 offset:8204
	s_mov_b32 s59, -1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v29, v30, v71, s4
	v_cndmask_b32_e64 v62, v30, v71, s7
	v_cndmask_b32_e64 v63, v30, v71, s10
	v_cndmask_b32_e64 v64, v30, v71, s13
	v_cndmask_b32_e64 v65, v30, v71, s16
	v_cndmask_b32_e64 v66, v30, v71, s19
	v_cndmask_b32_e64 v67, v30, v71, s22
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v29, v29, v72, s5
	v_cndmask_b32_e64 v68, v30, v71, s25
	v_cndmask_b32_e64 v69, v30, v71, s28
	v_cndmask_b32_e64 v62, v62, v72, s8
	v_cndmask_b32_e64 v77, v63, v72, s11
	v_cndmask_b32_e64 v64, v64, v72, s14
	v_cndmask_b32_e64 v78, v65, v72, s17
	v_cndmask_b32_e64 v66, v66, v72, s20
	v_cndmask_b32_e64 v79, v67, v72, s23
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v29, v29, v73, s6
	v_cndmask_b32_e64 v70, v30, v71, s31
	v_cndmask_b32_e64 v80, v68, v72, s26
	v_cndmask_b32_e64 v81, v69, v72, s29
	v_cndmask_b32_e64 v63, v62, v73, s9
	v_cndmask_b32_e64 v62, v77, v73, s12
	v_cndmask_b32_e64 v65, v64, v73, s15
	v_cndmask_b32_e64 v64, v78, v73, s18
	v_cndmask_b32_e64 v67, v66, v73, s21
	v_lshrrev_b64 v[68:69], v41, v[29:30]
	v_cndmask_b32_e64 v66, v79, v73, s24
	v_cndmask_b32_e64 v82, v70, v72, s33
	v_cndmask_b32_e64 v70, v80, v73, s27
	v_cndmask_b32_e64 v69, v81, v73, s30
	v_lshrrev_b64 v[62:63], v42, v[62:63]
	v_cndmask_b32_e64 v75, v30, v71, s38
	v_cndmask_b32_e64 v76, v30, v71, s41
	v_lshrrev_b64 v[63:64], v44, v[64:65]
	v_lshrrev_b64 v[64:65], v45, v[66:67]
	v_cndmask_b32_e64 v74, v30, v71, s35
	v_and_b32_e32 v68, 0xffff, v68
	v_lshrrev_b64 v[65:66], v46, v[69:70]
	v_cndmask_b32_e64 v29, v75, v72, s39
	v_and_b32_e32 v70, 0xffff, v62
	v_cndmask_b32_e64 v62, v30, v71, s44
	v_cndmask_b32_e64 v66, v30, v71, s47
	v_cndmask_b32_e64 v67, v76, v72, s42
	v_mul_lo_u32 v75, 0x83dcd12d, v68
	v_cndmask_b32_e64 v68, v74, v72, s36
	v_cndmask_b32_e64 v30, v29, v73, s40
	v_cndmask_b32_e64 v62, v62, v72, s45
	v_cndmask_b32_e64 v71, v66, v72, s48
	v_cndmask_b32_e64 v29, v67, v73, s43
	v_cndmask_b32_e64 v67, v82, v73, s34
	v_cndmask_b32_e64 v66, v68, v73, s37
	v_cndmask_b32_e64 v69, v62, v73, s46
	v_cndmask_b32_e64 v68, v71, v73, s49
	v_lshrrev_b64 v[29:30], v48, v[29:30]
	v_and_b32_e32 v30, 0xffff, v63
	v_lshrrev_b64 v[62:63], v47, v[66:67]
	v_and_b32_e32 v63, 0xffff, v65
	v_lshrrev_b64 v[65:66], v49, v[68:69]
	v_and_b32_e32 v64, 0xffff, v64
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_and_b32_e32 v62, 0xffff, v62
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	v_and_b32_e32 v65, 0xffff, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_mul_lo_u32 v66, 0x83dcd12d, v70
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_sad_u8 v67, v75, 0, 0x64006400
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_u8 v29, v29, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v66, v66, 0, v67
	v_sad_hi_u8 v30, v64, 0, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v62, v62, 0, v63
	v_sad_hi_u8 v29, v65, 0, v29
	ds_store_2addr_b32 v55, v66, v30 offset1:4
	ds_store_2addr_b32 v55, v62, v29 offset0:64 offset1:68
	; wave barrier
	s_branch .LBB10_34
.LBB10_31:                              ;   in Loop: Header=BB10_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[29:30], null, s58, s54, v[9:10]
	v_mov_b16_e32 v62.h, 0
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[29:30], 7, v[29:30]
	v_add_co_u32 v29, vcc_lo, v50, v29
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v30, null, v51, v30, vcc_lo
	global_load_b32 v29, v[29:30], off
	s_wait_loadcnt 0x0
	ds_bpermute_b32 v30, v61, v29
	v_bfe_u32 v63, v29, 12, 16
	v_bfe_u32 v67, v29, 4, 16
	v_mov_b16_e32 v62.l, v29.h
	v_bfe_u32 v64, v29, 8, 16
	v_and_b32_e32 v68, 0xffff, v29
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_hi_u8 v63, v64, 0, v63
	s_wait_dscnt 0x0
	v_alignbit_b32 v65, v30, v29, 28
	v_alignbit_b32 v66, v30, v29, 20
	v_alignbit_b32 v29, v30, v29, 24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v30, 0xffff, v65
	v_and_b32_e32 v65, 0xffff, v66
	v_mul_lo_u32 v66, 0x83dcd12d, v67
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v67, 0x83dcd12d, v68
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_sad_u8 v66, v66, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v65, v65, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sad_hi_u8 v64, v67, 0, v66
	v_sad_hi_u8 v29, v29, 0, v30
	s_delay_alu instid0(VALU_DEP_3)
	v_sad_hi_u8 v30, v62, 0, v65
	ds_store_2addr_b32 v55, v63, v64 offset0:64 offset1:68
	ds_store_2addr_b32 v55, v29, v30 offset1:4
	; wave barrier
	s_mov_b32 s52, -1
	s_mov_b32 s59, -1
	s_cbranch_execz .LBB10_2
	s_branch .LBB10_18
.LBB10_32:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s59, 0
	s_and_b32 vcc_lo, exec_lo, s52
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_34
; %bb.33:                               ;   in Loop: Header=BB10_3 Depth=1
	s_cmp_eq_u32 exec_lo, -1
	s_cbranch_scc1 .LBB10_49
.LBB10_34:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s52, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s59
	s_mov_b32 s59, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_2
.LBB10_35:                              ;   in Loop: Header=BB10_3 Depth=1
	ds_load_b128 v[62:65], v56
	ds_load_b128 v[66:69], v52 offset:64
	s_and_b32 vcc_lo, exec_lo, s65
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[1:8], v[62:65], v[66:69], v[1:8]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_41
; %bb.36:                               ;   in Loop: Header=BB10_3 Depth=1
	s_and_saveexec_b32 s59, s3
	s_cbranch_execz .LBB10_39
; %bb.37:                               ;   in Loop: Header=BB10_3 Depth=1
	v_dual_mov_b32 v30, v20 :: v_dual_mov_b32 v29, v19
	v_dual_mov_b32 v62, v58 :: v_dual_mov_b32 v63, v32
	s_mov_b32 s67, 0
.LBB10_38:                              ;   Parent Loop BB10_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_b32 v64, v[29:30], off
	v_add_nc_u32_e32 v63, 32, v63
	v_add_co_u32 v29, vcc_lo, 0x80, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, 0, v30, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s52, s66, v63
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s67, s52, s67
	s_wait_loadcnt 0x0
	ds_store_b32 v62, v64 offset:8192
	v_add_nc_u32_e32 v62, 0x80, v62
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s67
	s_cbranch_execnz .LBB10_38
.LBB10_39:                              ;   in Loop: Header=BB10_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s59
	; wave barrier
	ds_load_b32 v30, v37 offset:8192
	ds_load_b32 v71, v38 offset:8196
	ds_load_b32 v72, v39 offset:8200
	ds_load_b32 v73, v40 offset:8204
	s_mov_b32 s59, -1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v29, v30, v71, s4
	v_cndmask_b32_e64 v62, v30, v71, s7
	v_cndmask_b32_e64 v63, v30, v71, s10
	v_cndmask_b32_e64 v64, v30, v71, s13
	v_cndmask_b32_e64 v65, v30, v71, s16
	v_cndmask_b32_e64 v66, v30, v71, s19
	v_cndmask_b32_e64 v67, v30, v71, s22
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v29, v29, v72, s5
	v_cndmask_b32_e64 v68, v30, v71, s25
	v_cndmask_b32_e64 v69, v30, v71, s28
	v_cndmask_b32_e64 v62, v62, v72, s8
	v_cndmask_b32_e64 v77, v63, v72, s11
	v_cndmask_b32_e64 v64, v64, v72, s14
	v_cndmask_b32_e64 v78, v65, v72, s17
	v_cndmask_b32_e64 v66, v66, v72, s20
	v_cndmask_b32_e64 v79, v67, v72, s23
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v29, v29, v73, s6
	v_cndmask_b32_e64 v70, v30, v71, s31
	v_cndmask_b32_e64 v80, v68, v72, s26
	v_cndmask_b32_e64 v81, v69, v72, s29
	v_cndmask_b32_e64 v63, v62, v73, s9
	v_cndmask_b32_e64 v62, v77, v73, s12
	v_cndmask_b32_e64 v65, v64, v73, s15
	v_cndmask_b32_e64 v64, v78, v73, s18
	v_cndmask_b32_e64 v67, v66, v73, s21
	v_lshrrev_b64 v[68:69], v41, v[29:30]
	v_cndmask_b32_e64 v66, v79, v73, s24
	v_cndmask_b32_e64 v82, v70, v72, s33
	v_cndmask_b32_e64 v70, v80, v73, s27
	v_cndmask_b32_e64 v69, v81, v73, s30
	v_lshrrev_b64 v[62:63], v42, v[62:63]
	v_cndmask_b32_e64 v75, v30, v71, s38
	v_cndmask_b32_e64 v76, v30, v71, s41
	v_lshrrev_b64 v[63:64], v44, v[64:65]
	v_lshrrev_b64 v[64:65], v45, v[66:67]
	v_cndmask_b32_e64 v74, v30, v71, s35
	v_and_b32_e32 v68, 0xffff, v68
	v_lshrrev_b64 v[65:66], v46, v[69:70]
	v_cndmask_b32_e64 v29, v75, v72, s39
	v_and_b32_e32 v70, 0xffff, v62
	v_cndmask_b32_e64 v62, v30, v71, s44
	v_cndmask_b32_e64 v66, v30, v71, s47
	v_cndmask_b32_e64 v67, v76, v72, s42
	v_mul_lo_u32 v75, 0x83dcd12d, v68
	v_cndmask_b32_e64 v68, v74, v72, s36
	v_cndmask_b32_e64 v30, v29, v73, s40
	v_cndmask_b32_e64 v62, v62, v72, s45
	v_cndmask_b32_e64 v71, v66, v72, s48
	v_cndmask_b32_e64 v29, v67, v73, s43
	v_cndmask_b32_e64 v67, v82, v73, s34
	v_cndmask_b32_e64 v66, v68, v73, s37
	v_cndmask_b32_e64 v69, v62, v73, s46
	v_cndmask_b32_e64 v68, v71, v73, s49
	v_lshrrev_b64 v[29:30], v48, v[29:30]
	v_and_b32_e32 v30, 0xffff, v63
	v_lshrrev_b64 v[62:63], v47, v[66:67]
	v_and_b32_e32 v63, 0xffff, v65
	v_lshrrev_b64 v[65:66], v49, v[68:69]
	v_and_b32_e32 v64, 0xffff, v64
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_and_b32_e32 v62, 0xffff, v62
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	v_and_b32_e32 v65, 0xffff, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_mul_lo_u32 v66, 0x83dcd12d, v70
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_sad_u8 v67, v75, 0, 0x64006400
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_u8 v29, v29, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v66, v66, 0, v67
	v_sad_hi_u8 v30, v64, 0, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v62, v62, 0, v63
	v_sad_hi_u8 v29, v65, 0, v29
	ds_store_2addr_b32 v55, v66, v30 offset0:128 offset1:132
	ds_store_2addr_b32 v55, v62, v29 offset0:192 offset1:196
	; wave barrier
	s_branch .LBB10_43
.LBB10_40:                              ;   in Loop: Header=BB10_3 Depth=1
	s_or_b32 s52, s58, 1
	v_mov_b16_e32 v62.h, 0
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[29:30], null, s52, s54, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[29:30], 7, v[29:30]
	v_add_co_u32 v29, vcc_lo, v50, v29
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v30, null, v51, v30, vcc_lo
	global_load_b32 v29, v[29:30], off
	s_wait_loadcnt 0x0
	ds_bpermute_b32 v30, v61, v29
	v_bfe_u32 v63, v29, 12, 16
	v_bfe_u32 v67, v29, 4, 16
	v_mov_b16_e32 v62.l, v29.h
	v_bfe_u32 v64, v29, 8, 16
	v_and_b32_e32 v68, 0xffff, v29
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_hi_u8 v63, v64, 0, v63
	s_wait_dscnt 0x0
	v_alignbit_b32 v65, v30, v29, 28
	v_alignbit_b32 v66, v30, v29, 20
	v_alignbit_b32 v29, v30, v29, 24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v30, 0xffff, v65
	v_and_b32_e32 v65, 0xffff, v66
	v_mul_lo_u32 v66, 0x83dcd12d, v67
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v67, 0x83dcd12d, v68
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_sad_u8 v66, v66, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v65, v65, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sad_hi_u8 v64, v67, 0, v66
	v_sad_hi_u8 v29, v29, 0, v30
	s_delay_alu instid0(VALU_DEP_3)
	v_sad_hi_u8 v30, v62, 0, v65
	ds_store_2addr_b32 v55, v63, v64 offset0:192 offset1:196
	ds_store_2addr_b32 v55, v29, v30 offset0:128 offset1:132
	; wave barrier
	s_mov_b32 s52, -1
	s_mov_b32 s59, -1
	s_cbranch_execz .LBB10_2
	s_branch .LBB10_26
.LBB10_41:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s59, 0
	s_and_b32 vcc_lo, exec_lo, s52
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_43
; %bb.42:                               ;   in Loop: Header=BB10_3 Depth=1
	s_cmp_eq_u32 exec_lo, -1
	s_cbranch_scc1 .LBB10_58
.LBB10_43:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s52, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s59
	s_mov_b32 s59, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_2
.LBB10_44:                              ;   in Loop: Header=BB10_3 Depth=1
	ds_load_b128 v[62:65], v56 offset:512
	ds_load_b128 v[66:69], v52 offset:96
	s_and_b32 vcc_lo, exec_lo, s65
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[1:8], v[62:65], v[66:69], v[1:8]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_50
; %bb.45:                               ;   in Loop: Header=BB10_3 Depth=1
	s_and_saveexec_b32 s59, s3
	s_cbranch_execz .LBB10_48
; %bb.46:                               ;   in Loop: Header=BB10_3 Depth=1
	v_dual_mov_b32 v30, v22 :: v_dual_mov_b32 v29, v21
	v_dual_mov_b32 v62, v58 :: v_dual_mov_b32 v63, v32
	s_mov_b32 s67, 0
.LBB10_47:                              ;   Parent Loop BB10_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_b32 v64, v[29:30], off
	v_add_nc_u32_e32 v63, 32, v63
	v_add_co_u32 v29, vcc_lo, 0x80, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, 0, v30, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s52, s66, v63
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s67, s52, s67
	s_wait_loadcnt 0x0
	ds_store_b32 v62, v64 offset:8192
	v_add_nc_u32_e32 v62, 0x80, v62
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s67
	s_cbranch_execnz .LBB10_47
.LBB10_48:                              ;   in Loop: Header=BB10_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s59
	; wave barrier
	ds_load_b32 v30, v37 offset:8192
	ds_load_b32 v71, v38 offset:8196
	ds_load_b32 v72, v39 offset:8200
	ds_load_b32 v73, v40 offset:8204
	s_mov_b32 s59, -1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v29, v30, v71, s4
	v_cndmask_b32_e64 v62, v30, v71, s7
	v_cndmask_b32_e64 v63, v30, v71, s10
	v_cndmask_b32_e64 v64, v30, v71, s13
	v_cndmask_b32_e64 v65, v30, v71, s16
	v_cndmask_b32_e64 v66, v30, v71, s19
	v_cndmask_b32_e64 v67, v30, v71, s22
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v29, v29, v72, s5
	v_cndmask_b32_e64 v68, v30, v71, s25
	v_cndmask_b32_e64 v69, v30, v71, s28
	v_cndmask_b32_e64 v62, v62, v72, s8
	v_cndmask_b32_e64 v77, v63, v72, s11
	v_cndmask_b32_e64 v64, v64, v72, s14
	v_cndmask_b32_e64 v78, v65, v72, s17
	v_cndmask_b32_e64 v66, v66, v72, s20
	v_cndmask_b32_e64 v79, v67, v72, s23
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v29, v29, v73, s6
	v_cndmask_b32_e64 v70, v30, v71, s31
	v_cndmask_b32_e64 v80, v68, v72, s26
	v_cndmask_b32_e64 v81, v69, v72, s29
	v_cndmask_b32_e64 v63, v62, v73, s9
	v_cndmask_b32_e64 v62, v77, v73, s12
	v_cndmask_b32_e64 v65, v64, v73, s15
	v_cndmask_b32_e64 v64, v78, v73, s18
	v_cndmask_b32_e64 v67, v66, v73, s21
	v_lshrrev_b64 v[68:69], v41, v[29:30]
	v_cndmask_b32_e64 v66, v79, v73, s24
	v_cndmask_b32_e64 v82, v70, v72, s33
	v_cndmask_b32_e64 v70, v80, v73, s27
	v_cndmask_b32_e64 v69, v81, v73, s30
	v_lshrrev_b64 v[62:63], v42, v[62:63]
	v_cndmask_b32_e64 v75, v30, v71, s38
	v_cndmask_b32_e64 v76, v30, v71, s41
	v_lshrrev_b64 v[63:64], v44, v[64:65]
	v_lshrrev_b64 v[64:65], v45, v[66:67]
	v_cndmask_b32_e64 v74, v30, v71, s35
	v_and_b32_e32 v68, 0xffff, v68
	v_lshrrev_b64 v[65:66], v46, v[69:70]
	v_cndmask_b32_e64 v29, v75, v72, s39
	v_and_b32_e32 v70, 0xffff, v62
	v_cndmask_b32_e64 v62, v30, v71, s44
	v_cndmask_b32_e64 v66, v30, v71, s47
	v_cndmask_b32_e64 v67, v76, v72, s42
	v_mul_lo_u32 v75, 0x83dcd12d, v68
	v_cndmask_b32_e64 v68, v74, v72, s36
	v_cndmask_b32_e64 v30, v29, v73, s40
	v_cndmask_b32_e64 v62, v62, v72, s45
	v_cndmask_b32_e64 v71, v66, v72, s48
	v_cndmask_b32_e64 v29, v67, v73, s43
	v_cndmask_b32_e64 v67, v82, v73, s34
	v_cndmask_b32_e64 v66, v68, v73, s37
	v_cndmask_b32_e64 v69, v62, v73, s46
	v_cndmask_b32_e64 v68, v71, v73, s49
	v_lshrrev_b64 v[29:30], v48, v[29:30]
	v_and_b32_e32 v30, 0xffff, v63
	v_lshrrev_b64 v[62:63], v47, v[66:67]
	v_and_b32_e32 v63, 0xffff, v65
	v_lshrrev_b64 v[65:66], v49, v[68:69]
	v_and_b32_e32 v64, 0xffff, v64
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_and_b32_e32 v62, 0xffff, v62
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	v_and_b32_e32 v65, 0xffff, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_mul_lo_u32 v66, 0x83dcd12d, v70
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_sad_u8 v67, v75, 0, 0x64006400
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_u8 v29, v29, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v66, v66, 0, v67
	v_sad_hi_u8 v30, v64, 0, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v62, v62, 0, v63
	v_sad_hi_u8 v29, v65, 0, v29
	ds_store_2addr_b32 v55, v66, v30 offset1:4
	ds_store_2addr_b32 v55, v62, v29 offset0:64 offset1:68
	; wave barrier
	s_branch .LBB10_52
.LBB10_49:                              ;   in Loop: Header=BB10_3 Depth=1
	s_or_b32 s52, s58, 2
	v_mov_b16_e32 v62.h, 0
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[29:30], null, s52, s54, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[29:30], 7, v[29:30]
	v_add_co_u32 v29, vcc_lo, v50, v29
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v30, null, v51, v30, vcc_lo
	global_load_b32 v29, v[29:30], off
	s_wait_loadcnt 0x0
	ds_bpermute_b32 v30, v61, v29
	v_bfe_u32 v63, v29, 12, 16
	v_bfe_u32 v67, v29, 4, 16
	v_mov_b16_e32 v62.l, v29.h
	v_bfe_u32 v64, v29, 8, 16
	v_and_b32_e32 v68, 0xffff, v29
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_hi_u8 v63, v64, 0, v63
	s_wait_dscnt 0x0
	v_alignbit_b32 v65, v30, v29, 28
	v_alignbit_b32 v66, v30, v29, 20
	v_alignbit_b32 v29, v30, v29, 24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v30, 0xffff, v65
	v_and_b32_e32 v65, 0xffff, v66
	v_mul_lo_u32 v66, 0x83dcd12d, v67
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v67, 0x83dcd12d, v68
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_sad_u8 v66, v66, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v65, v65, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sad_hi_u8 v64, v67, 0, v66
	v_sad_hi_u8 v29, v29, 0, v30
	s_delay_alu instid0(VALU_DEP_3)
	v_sad_hi_u8 v30, v62, 0, v65
	ds_store_2addr_b32 v55, v63, v64 offset0:64 offset1:68
	ds_store_2addr_b32 v55, v29, v30 offset1:4
	; wave barrier
	s_mov_b32 s52, -1
	s_mov_b32 s59, -1
	s_cbranch_execz .LBB10_2
	s_branch .LBB10_35
.LBB10_50:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s59, 0
	s_and_b32 vcc_lo, exec_lo, s52
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_52
; %bb.51:                               ;   in Loop: Header=BB10_3 Depth=1
	s_cmp_eq_u32 exec_lo, -1
	s_cbranch_scc1 .LBB10_68
.LBB10_52:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s52, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s59
	s_mov_b32 s59, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_2
.LBB10_53:                              ;   in Loop: Header=BB10_3 Depth=1
	ds_load_b128 v[62:65], v56
	ds_load_b128 v[66:69], v52 offset:128
	s_and_b32 vcc_lo, exec_lo, s65
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[1:8], v[62:65], v[66:69], v[1:8]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_59
; %bb.54:                               ;   in Loop: Header=BB10_3 Depth=1
	s_and_saveexec_b32 s59, s3
	s_cbranch_execz .LBB10_57
; %bb.55:                               ;   in Loop: Header=BB10_3 Depth=1
	v_dual_mov_b32 v30, v24 :: v_dual_mov_b32 v29, v23
	v_dual_mov_b32 v62, v58 :: v_dual_mov_b32 v63, v32
	s_mov_b32 s67, 0
.LBB10_56:                              ;   Parent Loop BB10_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_b32 v64, v[29:30], off
	v_add_nc_u32_e32 v63, 32, v63
	v_add_co_u32 v29, vcc_lo, 0x80, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, 0, v30, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s52, s66, v63
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s67, s52, s67
	s_wait_loadcnt 0x0
	ds_store_b32 v62, v64 offset:8192
	v_add_nc_u32_e32 v62, 0x80, v62
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s67
	s_cbranch_execnz .LBB10_56
.LBB10_57:                              ;   in Loop: Header=BB10_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s59
	; wave barrier
	ds_load_b32 v30, v37 offset:8192
	ds_load_b32 v71, v38 offset:8196
	ds_load_b32 v72, v39 offset:8200
	ds_load_b32 v73, v40 offset:8204
	s_mov_b32 s59, -1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v29, v30, v71, s4
	v_cndmask_b32_e64 v62, v30, v71, s7
	v_cndmask_b32_e64 v63, v30, v71, s10
	v_cndmask_b32_e64 v64, v30, v71, s13
	v_cndmask_b32_e64 v65, v30, v71, s16
	v_cndmask_b32_e64 v66, v30, v71, s19
	v_cndmask_b32_e64 v67, v30, v71, s22
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v29, v29, v72, s5
	v_cndmask_b32_e64 v68, v30, v71, s25
	v_cndmask_b32_e64 v69, v30, v71, s28
	v_cndmask_b32_e64 v62, v62, v72, s8
	v_cndmask_b32_e64 v77, v63, v72, s11
	v_cndmask_b32_e64 v64, v64, v72, s14
	v_cndmask_b32_e64 v78, v65, v72, s17
	v_cndmask_b32_e64 v66, v66, v72, s20
	v_cndmask_b32_e64 v79, v67, v72, s23
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v29, v29, v73, s6
	v_cndmask_b32_e64 v70, v30, v71, s31
	v_cndmask_b32_e64 v80, v68, v72, s26
	v_cndmask_b32_e64 v81, v69, v72, s29
	v_cndmask_b32_e64 v63, v62, v73, s9
	v_cndmask_b32_e64 v62, v77, v73, s12
	v_cndmask_b32_e64 v65, v64, v73, s15
	v_cndmask_b32_e64 v64, v78, v73, s18
	v_cndmask_b32_e64 v67, v66, v73, s21
	v_lshrrev_b64 v[68:69], v41, v[29:30]
	v_cndmask_b32_e64 v66, v79, v73, s24
	v_cndmask_b32_e64 v82, v70, v72, s33
	v_cndmask_b32_e64 v70, v80, v73, s27
	v_cndmask_b32_e64 v69, v81, v73, s30
	v_lshrrev_b64 v[62:63], v42, v[62:63]
	v_cndmask_b32_e64 v75, v30, v71, s38
	v_cndmask_b32_e64 v76, v30, v71, s41
	v_lshrrev_b64 v[63:64], v44, v[64:65]
	v_lshrrev_b64 v[64:65], v45, v[66:67]
	v_cndmask_b32_e64 v74, v30, v71, s35
	v_and_b32_e32 v68, 0xffff, v68
	v_lshrrev_b64 v[65:66], v46, v[69:70]
	v_cndmask_b32_e64 v29, v75, v72, s39
	v_and_b32_e32 v70, 0xffff, v62
	v_cndmask_b32_e64 v62, v30, v71, s44
	v_cndmask_b32_e64 v66, v30, v71, s47
	v_cndmask_b32_e64 v67, v76, v72, s42
	v_mul_lo_u32 v75, 0x83dcd12d, v68
	v_cndmask_b32_e64 v68, v74, v72, s36
	v_cndmask_b32_e64 v30, v29, v73, s40
	v_cndmask_b32_e64 v62, v62, v72, s45
	v_cndmask_b32_e64 v71, v66, v72, s48
	v_cndmask_b32_e64 v29, v67, v73, s43
	v_cndmask_b32_e64 v67, v82, v73, s34
	v_cndmask_b32_e64 v66, v68, v73, s37
	v_cndmask_b32_e64 v69, v62, v73, s46
	v_cndmask_b32_e64 v68, v71, v73, s49
	v_lshrrev_b64 v[29:30], v48, v[29:30]
	v_and_b32_e32 v30, 0xffff, v63
	v_lshrrev_b64 v[62:63], v47, v[66:67]
	v_and_b32_e32 v63, 0xffff, v65
	v_lshrrev_b64 v[65:66], v49, v[68:69]
	v_and_b32_e32 v64, 0xffff, v64
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_and_b32_e32 v62, 0xffff, v62
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	v_and_b32_e32 v65, 0xffff, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_mul_lo_u32 v66, 0x83dcd12d, v70
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_sad_u8 v67, v75, 0, 0x64006400
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_u8 v29, v29, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v66, v66, 0, v67
	v_sad_hi_u8 v30, v64, 0, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v62, v62, 0, v63
	v_sad_hi_u8 v29, v65, 0, v29
	ds_store_2addr_b32 v55, v66, v30 offset0:128 offset1:132
	ds_store_2addr_b32 v55, v62, v29 offset0:192 offset1:196
	; wave barrier
	s_branch .LBB10_62
.LBB10_58:                              ;   in Loop: Header=BB10_3 Depth=1
	s_or_b32 s52, s58, 3
	v_mov_b16_e32 v62.h, 0
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[29:30], null, s52, s54, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[29:30], 7, v[29:30]
	v_add_co_u32 v29, vcc_lo, v50, v29
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v30, null, v51, v30, vcc_lo
	global_load_b32 v29, v[29:30], off
	s_wait_loadcnt 0x0
	ds_bpermute_b32 v30, v61, v29
	v_bfe_u32 v63, v29, 12, 16
	v_bfe_u32 v67, v29, 4, 16
	v_mov_b16_e32 v62.l, v29.h
	v_bfe_u32 v64, v29, 8, 16
	v_and_b32_e32 v68, 0xffff, v29
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_hi_u8 v63, v64, 0, v63
	s_wait_dscnt 0x0
	v_alignbit_b32 v65, v30, v29, 28
	v_alignbit_b32 v66, v30, v29, 20
	v_alignbit_b32 v29, v30, v29, 24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v30, 0xffff, v65
	v_and_b32_e32 v65, 0xffff, v66
	v_mul_lo_u32 v66, 0x83dcd12d, v67
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v67, 0x83dcd12d, v68
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_sad_u8 v66, v66, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v65, v65, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sad_hi_u8 v64, v67, 0, v66
	v_sad_hi_u8 v29, v29, 0, v30
	s_delay_alu instid0(VALU_DEP_3)
	v_sad_hi_u8 v30, v62, 0, v65
	ds_store_2addr_b32 v55, v63, v64 offset0:192 offset1:196
	ds_store_2addr_b32 v55, v29, v30 offset0:128 offset1:132
	; wave barrier
	s_mov_b32 s52, -1
	s_mov_b32 s59, -1
	s_cbranch_execz .LBB10_2
	s_branch .LBB10_44
.LBB10_59:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s59, 0
	s_and_b32 vcc_lo, exec_lo, s52
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_62
; %bb.60:                               ;   in Loop: Header=BB10_3 Depth=1
	s_cmp_eq_u32 exec_lo, -1
	s_cbranch_scc0 .LBB10_62
; %bb.61:                               ;   in Loop: Header=BB10_3 Depth=1
	s_or_b32 s52, s58, 5
	v_mov_b16_e32 v62.h, 0
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[29:30], null, s52, s54, v[9:10]
	s_mov_b32 s59, -1
	v_lshlrev_b64_e32 v[29:30], 7, v[29:30]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, v50, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, v51, v30, vcc_lo
	global_load_b32 v29, v[29:30], off
	s_wait_loadcnt 0x0
	ds_bpermute_b32 v30, v61, v29
	v_bfe_u32 v63, v29, 12, 16
	v_bfe_u32 v67, v29, 4, 16
	v_mov_b16_e32 v62.l, v29.h
	v_bfe_u32 v64, v29, 8, 16
	v_and_b32_e32 v68, 0xffff, v29
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_hi_u8 v63, v64, 0, v63
	s_wait_dscnt 0x0
	v_alignbit_b32 v65, v30, v29, 28
	v_alignbit_b32 v66, v30, v29, 20
	v_alignbit_b32 v29, v30, v29, 24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v30, 0xffff, v65
	v_and_b32_e32 v65, 0xffff, v66
	v_mul_lo_u32 v66, 0x83dcd12d, v67
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v67, 0x83dcd12d, v68
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_sad_u8 v66, v66, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v65, v65, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sad_hi_u8 v64, v67, 0, v66
	v_sad_hi_u8 v29, v29, 0, v30
	s_delay_alu instid0(VALU_DEP_3)
	v_sad_hi_u8 v30, v62, 0, v65
	ds_store_2addr_b32 v55, v63, v64 offset0:192 offset1:196
	ds_store_2addr_b32 v55, v29, v30 offset0:128 offset1:132
	; wave barrier
.LBB10_62:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s52, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s59
	s_mov_b32 s59, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_2
; %bb.63:                               ;   in Loop: Header=BB10_3 Depth=1
	ds_load_b128 v[62:65], v56 offset:512
	ds_load_b128 v[66:69], v52 offset:160
	s_and_b32 vcc_lo, exec_lo, s65
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[1:8], v[62:65], v[66:69], v[1:8]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_69
; %bb.64:                               ;   in Loop: Header=BB10_3 Depth=1
	s_and_saveexec_b32 s59, s3
	s_cbranch_execz .LBB10_67
; %bb.65:                               ;   in Loop: Header=BB10_3 Depth=1
	v_dual_mov_b32 v30, v26 :: v_dual_mov_b32 v29, v25
	v_dual_mov_b32 v62, v58 :: v_dual_mov_b32 v63, v32
	s_mov_b32 s67, 0
.LBB10_66:                              ;   Parent Loop BB10_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_b32 v64, v[29:30], off
	v_add_nc_u32_e32 v63, 32, v63
	v_add_co_u32 v29, vcc_lo, 0x80, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, 0, v30, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s52, s66, v63
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s67, s52, s67
	s_wait_loadcnt 0x0
	ds_store_b32 v62, v64 offset:8192
	v_add_nc_u32_e32 v62, 0x80, v62
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s67
	s_cbranch_execnz .LBB10_66
.LBB10_67:                              ;   in Loop: Header=BB10_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s59
	; wave barrier
	ds_load_b32 v30, v37 offset:8192
	ds_load_b32 v71, v38 offset:8196
	ds_load_b32 v72, v39 offset:8200
	ds_load_b32 v73, v40 offset:8204
	s_mov_b32 s59, -1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v29, v30, v71, s4
	v_cndmask_b32_e64 v62, v30, v71, s7
	v_cndmask_b32_e64 v63, v30, v71, s10
	v_cndmask_b32_e64 v64, v30, v71, s13
	v_cndmask_b32_e64 v65, v30, v71, s16
	v_cndmask_b32_e64 v66, v30, v71, s19
	v_cndmask_b32_e64 v67, v30, v71, s22
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v29, v29, v72, s5
	v_cndmask_b32_e64 v68, v30, v71, s25
	v_cndmask_b32_e64 v69, v30, v71, s28
	v_cndmask_b32_e64 v62, v62, v72, s8
	v_cndmask_b32_e64 v77, v63, v72, s11
	v_cndmask_b32_e64 v64, v64, v72, s14
	v_cndmask_b32_e64 v78, v65, v72, s17
	v_cndmask_b32_e64 v66, v66, v72, s20
	v_cndmask_b32_e64 v79, v67, v72, s23
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v29, v29, v73, s6
	v_cndmask_b32_e64 v70, v30, v71, s31
	v_cndmask_b32_e64 v80, v68, v72, s26
	v_cndmask_b32_e64 v81, v69, v72, s29
	v_cndmask_b32_e64 v63, v62, v73, s9
	v_cndmask_b32_e64 v62, v77, v73, s12
	v_cndmask_b32_e64 v65, v64, v73, s15
	v_cndmask_b32_e64 v64, v78, v73, s18
	v_cndmask_b32_e64 v67, v66, v73, s21
	v_lshrrev_b64 v[68:69], v41, v[29:30]
	v_cndmask_b32_e64 v66, v79, v73, s24
	v_cndmask_b32_e64 v82, v70, v72, s33
	v_cndmask_b32_e64 v70, v80, v73, s27
	v_cndmask_b32_e64 v69, v81, v73, s30
	v_lshrrev_b64 v[62:63], v42, v[62:63]
	v_cndmask_b32_e64 v75, v30, v71, s38
	v_cndmask_b32_e64 v76, v30, v71, s41
	v_lshrrev_b64 v[63:64], v44, v[64:65]
	v_lshrrev_b64 v[64:65], v45, v[66:67]
	v_cndmask_b32_e64 v74, v30, v71, s35
	v_and_b32_e32 v68, 0xffff, v68
	v_lshrrev_b64 v[65:66], v46, v[69:70]
	v_cndmask_b32_e64 v29, v75, v72, s39
	v_and_b32_e32 v70, 0xffff, v62
	v_cndmask_b32_e64 v62, v30, v71, s44
	v_cndmask_b32_e64 v66, v30, v71, s47
	v_cndmask_b32_e64 v67, v76, v72, s42
	v_mul_lo_u32 v75, 0x83dcd12d, v68
	v_cndmask_b32_e64 v68, v74, v72, s36
	v_cndmask_b32_e64 v30, v29, v73, s40
	v_cndmask_b32_e64 v62, v62, v72, s45
	v_cndmask_b32_e64 v71, v66, v72, s48
	v_cndmask_b32_e64 v29, v67, v73, s43
	v_cndmask_b32_e64 v67, v82, v73, s34
	v_cndmask_b32_e64 v66, v68, v73, s37
	v_cndmask_b32_e64 v69, v62, v73, s46
	v_cndmask_b32_e64 v68, v71, v73, s49
	v_lshrrev_b64 v[29:30], v48, v[29:30]
	v_and_b32_e32 v30, 0xffff, v63
	v_lshrrev_b64 v[62:63], v47, v[66:67]
	v_and_b32_e32 v63, 0xffff, v65
	v_lshrrev_b64 v[65:66], v49, v[68:69]
	v_and_b32_e32 v64, 0xffff, v64
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_and_b32_e32 v62, 0xffff, v62
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	v_and_b32_e32 v65, 0xffff, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_mul_lo_u32 v66, 0x83dcd12d, v70
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_sad_u8 v67, v75, 0, 0x64006400
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_u8 v29, v29, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v66, v66, 0, v67
	v_sad_hi_u8 v30, v64, 0, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v62, v62, 0, v63
	v_sad_hi_u8 v29, v65, 0, v29
	ds_store_2addr_b32 v55, v66, v30 offset1:4
	ds_store_2addr_b32 v55, v62, v29 offset0:64 offset1:68
	; wave barrier
	s_branch .LBB10_72
.LBB10_68:                              ;   in Loop: Header=BB10_3 Depth=1
	s_or_b32 s52, s58, 4
	v_mov_b16_e32 v62.h, 0
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[29:30], null, s52, s54, v[9:10]
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_lshlrev_b64_e32 v[29:30], 7, v[29:30]
	v_add_co_u32 v29, vcc_lo, v50, v29
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_2)
	v_add_co_ci_u32_e64 v30, null, v51, v30, vcc_lo
	global_load_b32 v29, v[29:30], off
	s_wait_loadcnt 0x0
	ds_bpermute_b32 v30, v61, v29
	v_bfe_u32 v63, v29, 12, 16
	v_bfe_u32 v67, v29, 4, 16
	v_mov_b16_e32 v62.l, v29.h
	v_bfe_u32 v64, v29, 8, 16
	v_and_b32_e32 v68, 0xffff, v29
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_hi_u8 v63, v64, 0, v63
	s_wait_dscnt 0x0
	v_alignbit_b32 v65, v30, v29, 28
	v_alignbit_b32 v66, v30, v29, 20
	v_alignbit_b32 v29, v30, v29, 24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v30, 0xffff, v65
	v_and_b32_e32 v65, 0xffff, v66
	v_mul_lo_u32 v66, 0x83dcd12d, v67
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v67, 0x83dcd12d, v68
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_sad_u8 v66, v66, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v65, v65, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sad_hi_u8 v64, v67, 0, v66
	v_sad_hi_u8 v29, v29, 0, v30
	s_delay_alu instid0(VALU_DEP_3)
	v_sad_hi_u8 v30, v62, 0, v65
	ds_store_2addr_b32 v55, v63, v64 offset0:64 offset1:68
	ds_store_2addr_b32 v55, v29, v30 offset1:4
	; wave barrier
	s_mov_b32 s52, -1
	s_mov_b32 s59, -1
	s_cbranch_execz .LBB10_2
	s_branch .LBB10_53
.LBB10_69:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s59, 0
	s_and_b32 vcc_lo, exec_lo, s52
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_72
; %bb.70:                               ;   in Loop: Header=BB10_3 Depth=1
	s_cmp_eq_u32 exec_lo, -1
	s_cbranch_scc0 .LBB10_72
; %bb.71:                               ;   in Loop: Header=BB10_3 Depth=1
	s_or_b32 s52, s58, 6
	v_mov_b16_e32 v62.h, 0
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[29:30], null, s52, s54, v[9:10]
	s_mov_b32 s59, -1
	v_lshlrev_b64_e32 v[29:30], 7, v[29:30]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, v50, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, v51, v30, vcc_lo
	global_load_b32 v29, v[29:30], off
	s_wait_loadcnt 0x0
	ds_bpermute_b32 v30, v61, v29
	v_bfe_u32 v63, v29, 12, 16
	v_bfe_u32 v67, v29, 4, 16
	v_mov_b16_e32 v62.l, v29.h
	v_bfe_u32 v64, v29, 8, 16
	v_and_b32_e32 v68, 0xffff, v29
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_hi_u8 v63, v64, 0, v63
	s_wait_dscnt 0x0
	v_alignbit_b32 v65, v30, v29, 28
	v_alignbit_b32 v66, v30, v29, 20
	v_alignbit_b32 v29, v30, v29, 24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v30, 0xffff, v65
	v_and_b32_e32 v65, 0xffff, v66
	v_mul_lo_u32 v66, 0x83dcd12d, v67
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v67, 0x83dcd12d, v68
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_sad_u8 v66, v66, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v65, v65, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sad_hi_u8 v64, v67, 0, v66
	v_sad_hi_u8 v29, v29, 0, v30
	s_delay_alu instid0(VALU_DEP_3)
	v_sad_hi_u8 v30, v62, 0, v65
	ds_store_2addr_b32 v55, v63, v64 offset0:64 offset1:68
	ds_store_2addr_b32 v55, v29, v30 offset1:4
	; wave barrier
.LBB10_72:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s52, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s59
	s_mov_b32 s59, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_2
; %bb.73:                               ;   in Loop: Header=BB10_3 Depth=1
	ds_load_b128 v[62:65], v56
	ds_load_b128 v[66:69], v52 offset:192
	s_and_b32 vcc_lo, exec_lo, s65
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[1:8], v[62:65], v[66:69], v[1:8]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_78
; %bb.74:                               ;   in Loop: Header=BB10_3 Depth=1
	s_and_saveexec_b32 s59, s3
	s_cbranch_execz .LBB10_77
; %bb.75:                               ;   in Loop: Header=BB10_3 Depth=1
	v_dual_mov_b32 v30, v28 :: v_dual_mov_b32 v29, v27
	v_dual_mov_b32 v62, v58 :: v_dual_mov_b32 v63, v32
	s_mov_b32 s67, 0
.LBB10_76:                              ;   Parent Loop BB10_3 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	global_load_b32 v64, v[29:30], off
	v_add_nc_u32_e32 v63, 32, v63
	v_add_co_u32 v29, vcc_lo, 0x80, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, 0, v30, vcc_lo
	s_delay_alu instid0(VALU_DEP_3)
	v_cmp_le_u32_e64 s52, s66, v63
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 s67, s52, s67
	s_wait_loadcnt 0x0
	ds_store_b32 v62, v64 offset:8192
	v_add_nc_u32_e32 v62, 0x80, v62
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_b32 exec_lo, exec_lo, s67
	s_cbranch_execnz .LBB10_76
.LBB10_77:                              ;   in Loop: Header=BB10_3 Depth=1
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s59
	; wave barrier
	ds_load_b32 v30, v37 offset:8192
	ds_load_b32 v71, v38 offset:8196
	ds_load_b32 v72, v39 offset:8200
	ds_load_b32 v73, v40 offset:8204
	s_mov_b32 s59, -1
	s_wait_dscnt 0x2
	v_cndmask_b32_e64 v29, v30, v71, s4
	v_cndmask_b32_e64 v62, v30, v71, s7
	v_cndmask_b32_e64 v63, v30, v71, s10
	v_cndmask_b32_e64 v64, v30, v71, s13
	v_cndmask_b32_e64 v65, v30, v71, s16
	v_cndmask_b32_e64 v66, v30, v71, s19
	v_cndmask_b32_e64 v67, v30, v71, s22
	s_wait_dscnt 0x1
	v_cndmask_b32_e64 v29, v29, v72, s5
	v_cndmask_b32_e64 v68, v30, v71, s25
	v_cndmask_b32_e64 v69, v30, v71, s28
	v_cndmask_b32_e64 v62, v62, v72, s8
	v_cndmask_b32_e64 v77, v63, v72, s11
	v_cndmask_b32_e64 v64, v64, v72, s14
	v_cndmask_b32_e64 v78, v65, v72, s17
	v_cndmask_b32_e64 v66, v66, v72, s20
	v_cndmask_b32_e64 v79, v67, v72, s23
	s_wait_dscnt 0x0
	v_cndmask_b32_e64 v29, v29, v73, s6
	v_cndmask_b32_e64 v70, v30, v71, s31
	v_cndmask_b32_e64 v80, v68, v72, s26
	v_cndmask_b32_e64 v81, v69, v72, s29
	v_cndmask_b32_e64 v63, v62, v73, s9
	v_cndmask_b32_e64 v62, v77, v73, s12
	v_cndmask_b32_e64 v65, v64, v73, s15
	v_cndmask_b32_e64 v64, v78, v73, s18
	v_cndmask_b32_e64 v67, v66, v73, s21
	v_lshrrev_b64 v[68:69], v41, v[29:30]
	v_cndmask_b32_e64 v66, v79, v73, s24
	v_cndmask_b32_e64 v82, v70, v72, s33
	v_cndmask_b32_e64 v70, v80, v73, s27
	v_cndmask_b32_e64 v69, v81, v73, s30
	v_lshrrev_b64 v[62:63], v42, v[62:63]
	v_cndmask_b32_e64 v75, v30, v71, s38
	v_cndmask_b32_e64 v76, v30, v71, s41
	v_lshrrev_b64 v[63:64], v44, v[64:65]
	v_lshrrev_b64 v[64:65], v45, v[66:67]
	v_cndmask_b32_e64 v74, v30, v71, s35
	v_and_b32_e32 v68, 0xffff, v68
	v_lshrrev_b64 v[65:66], v46, v[69:70]
	v_cndmask_b32_e64 v29, v75, v72, s39
	v_and_b32_e32 v70, 0xffff, v62
	v_cndmask_b32_e64 v62, v30, v71, s44
	v_cndmask_b32_e64 v66, v30, v71, s47
	v_cndmask_b32_e64 v67, v76, v72, s42
	v_mul_lo_u32 v75, 0x83dcd12d, v68
	v_cndmask_b32_e64 v68, v74, v72, s36
	v_cndmask_b32_e64 v30, v29, v73, s40
	v_cndmask_b32_e64 v62, v62, v72, s45
	v_cndmask_b32_e64 v71, v66, v72, s48
	v_cndmask_b32_e64 v29, v67, v73, s43
	v_cndmask_b32_e64 v67, v82, v73, s34
	v_cndmask_b32_e64 v66, v68, v73, s37
	v_cndmask_b32_e64 v69, v62, v73, s46
	v_cndmask_b32_e64 v68, v71, v73, s49
	v_lshrrev_b64 v[29:30], v48, v[29:30]
	v_and_b32_e32 v30, 0xffff, v63
	v_lshrrev_b64 v[62:63], v47, v[66:67]
	v_and_b32_e32 v63, 0xffff, v65
	v_lshrrev_b64 v[65:66], v49, v[68:69]
	v_and_b32_e32 v64, 0xffff, v64
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_and_b32_e32 v62, 0xffff, v62
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	v_and_b32_e32 v65, 0xffff, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_mul_lo_u32 v66, 0x83dcd12d, v70
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_sad_u8 v67, v75, 0, 0x64006400
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_u8 v29, v29, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v66, v66, 0, v67
	v_sad_hi_u8 v30, v64, 0, v30
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_hi_u8 v62, v62, 0, v63
	v_sad_hi_u8 v29, v65, 0, v29
	ds_store_2addr_b32 v55, v66, v30 offset0:128 offset1:132
	ds_store_2addr_b32 v55, v62, v29 offset0:192 offset1:196
	; wave barrier
	s_branch .LBB10_81
.LBB10_78:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s59, 0
	s_and_b32 vcc_lo, exec_lo, s52
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_81
; %bb.79:                               ;   in Loop: Header=BB10_3 Depth=1
	s_cmp_eq_u32 exec_lo, -1
	s_cbranch_scc0 .LBB10_81
; %bb.80:                               ;   in Loop: Header=BB10_3 Depth=1
	s_or_b32 s58, s58, 7
	v_mov_b16_e32 v62.h, 0
	s_wait_alu depctr_sa_sdst(0)
	v_mad_co_u64_u32 v[29:30], null, s58, s54, v[9:10]
	s_mov_b32 s59, -1
	v_lshlrev_b64_e32 v[29:30], 7, v[29:30]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v29, vcc_lo, v50, v29
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v30, null, v51, v30, vcc_lo
	global_load_b32 v29, v[29:30], off
	s_wait_loadcnt 0x0
	ds_bpermute_b32 v30, v61, v29
	v_bfe_u32 v63, v29, 12, 16
	v_bfe_u32 v67, v29, 4, 16
	v_mov_b16_e32 v62.l, v29.h
	v_bfe_u32 v64, v29, 8, 16
	v_and_b32_e32 v68, 0xffff, v29
	v_mul_lo_u32 v63, 0x83dcd12d, v63
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_mul_lo_u32 v62, 0x83dcd12d, v62
	v_mul_lo_u32 v64, 0x83dcd12d, v64
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_sad_u8 v63, v63, 0, 0x64006400
	v_sad_hi_u8 v63, v64, 0, v63
	s_wait_dscnt 0x0
	v_alignbit_b32 v65, v30, v29, 28
	v_alignbit_b32 v66, v30, v29, 20
	v_alignbit_b32 v29, v30, v29, 24
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_and_b32_e32 v30, 0xffff, v65
	v_and_b32_e32 v65, 0xffff, v66
	v_mul_lo_u32 v66, 0x83dcd12d, v67
	s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)
	v_and_b32_e32 v29, 0xffff, v29
	v_mul_lo_u32 v67, 0x83dcd12d, v68
	v_mul_lo_u32 v30, 0x83dcd12d, v30
	v_mul_lo_u32 v65, 0x83dcd12d, v65
	v_mul_lo_u32 v29, 0x83dcd12d, v29
	v_sad_u8 v66, v66, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
	v_sad_u8 v30, v30, 0, 0x64006400
	v_sad_u8 v65, v65, 0, 0x64006400
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)
	v_sad_hi_u8 v64, v67, 0, v66
	v_sad_hi_u8 v29, v29, 0, v30
	s_delay_alu instid0(VALU_DEP_3)
	v_sad_hi_u8 v30, v62, 0, v65
	ds_store_2addr_b32 v55, v63, v64 offset0:192 offset1:196
	ds_store_2addr_b32 v55, v29, v30 offset0:128 offset1:132
	; wave barrier
.LBB10_81:                              ;   in Loop: Header=BB10_3 Depth=1
	s_mov_b32 s52, -1
	s_wait_alu depctr_sa_sdst(0)
	s_and_b32 vcc_lo, exec_lo, s59
	s_mov_b32 s59, -1
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_2
; %bb.82:                               ;   in Loop: Header=BB10_3 Depth=1
	ds_load_b128 v[62:65], v56 offset:512
	ds_load_b128 v[66:69], v52 offset:224
	v_add_co_u32 v13, vcc_lo, v13, s56
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v14, null, s57, v14, vcc_lo
	v_add_co_u32 v15, vcc_lo, v15, s56
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, s57, v16, vcc_lo
	v_add_co_u32 v17, vcc_lo, v17, s56
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v18, null, s57, v18, vcc_lo
	v_add_co_u32 v19, vcc_lo, v19, s56
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v20, null, s57, v20, vcc_lo
	v_add_co_u32 v21, vcc_lo, v21, s56
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v22, null, s57, v22, vcc_lo
	v_add_co_u32 v23, vcc_lo, v23, s56
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v24, null, s57, v24, vcc_lo
	v_add_co_u32 v25, vcc_lo, v25, s56
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v26, null, s57, v26, vcc_lo
	v_add_co_u32 v27, vcc_lo, v27, s56
	s_add_co_i32 s55, s55, 1
	s_wait_dscnt 0x0
	v_wmma_f32_16x16x16_f16 v[1:8], v[62:65], v[66:69], v[1:8]
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v28, null, s57, v28, vcc_lo
	s_wait_alu depctr_sa_sdst(0)
	s_cmp_eq_u32 s55, s63
	s_mov_b32 s52, 0
	s_cselect_b32 s59, -1, 0
	s_branch .LBB10_2
.LBB10_83:
	s_xor_b32 s2, s52, -1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_and_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_143
; %bb.84:
	s_mov_b32 s2, -1
	s_cbranch_execz .LBB10_144
; %bb.85:
	s_branch .LBB10_87
.LBB10_86:
	v_dual_mov_b32 v8, 0 :: v_dual_mov_b32 v31, v36
	s_mov_b32 s2, -1
	s_delay_alu instid0(VALU_DEP_1)
	v_dual_mov_b32 v7, v8 :: v_dual_mov_b32 v6, v8
	v_dual_mov_b32 v5, v8 :: v_dual_mov_b32 v4, v8
	v_dual_mov_b32 v3, v8 :: v_dual_mov_b32 v2, v8
	v_dual_mov_b32 v1, v8 :: v_dual_mov_b32 v12, v8
	v_mov_b32_e32 v11, v8
.LBB10_87:
	s_and_not1_b32 vcc_lo, exec_lo, s2
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB10_141
; %bb.88:
	ds_swizzle_b32 v10, v11 offset:swizzle(BITMASK_PERM,"1pppp")
	v_mbcnt_lo_u32_b32 v14, -1, 0
	s_clause 0x1
	s_load_b64 s[10:11], s[0:1], 0x10
	s_load_b64 s[8:9], s[0:1], 0x38
	v_cmp_eq_u32_e64 s2, 0, v32
	v_cmp_lt_u32_e32 vcc_lo, 23, v14
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v9, 8, 0, vcc_lo
	v_cmp_lt_u32_e32 vcc_lo, 27, v14
	s_delay_alu instid0(VALU_DEP_2)
	v_add_nc_u32_e32 v9, v9, v14
	s_wait_dscnt 0x0
	v_add_f32_e32 v11, v11, v10
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v10, 4, 0, vcc_lo
	v_lshlrev_b32_e32 v9, 2, v9
	v_cmp_lt_u32_e32 vcc_lo, 29, v14
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)
	v_add_nc_u32_e32 v10, v10, v14
	v_lshlrev_b32_e32 v10, 2, v10
	ds_bpermute_b32 v13, v9, v11
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v11, v13
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e64 v11, 2, 0, vcc_lo
	v_cmp_ne_u32_e32 vcc_lo, 31, v14
	ds_bpermute_b32 v15, v10, v13
	v_add_nc_u32_e32 v11, v11, v14
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v16, null, 0, v14, vcc_lo
	v_cmp_gt_u32_e32 vcc_lo, s61, v31
	s_delay_alu instid0(VALU_DEP_3)
	v_lshlrev_b32_e32 v11, 2, v11
	s_and_b32 s4, s2, vcc_lo
	s_wait_dscnt 0x0
	v_add_f32_e32 v13, v13, v15
	ds_bpermute_b32 v15, v11, v13
	s_wait_dscnt 0x0
	v_add_f32_e32 v15, v13, v15
	v_lshlrev_b32_e32 v13, 2, v16
	ds_bpermute_b32 v16, v13, v15
	s_wait_alu depctr_sa_sdst(0)
	s_and_saveexec_b32 s3, s4
	s_cbranch_execz .LBB10_90
; %bb.89:
	s_wait_dscnt 0x0
	v_dual_add_f32 v15, v15, v16 :: v_dual_lshlrev_b32 v16, 2, v31
	ds_store_b32 v16, v15 offset:14336
.LBB10_90:
	s_or_b32 exec_lo, exec_lo, s3
	ds_swizzle_b32 v15, v12 offset:swizzle(BITMASK_PERM,"1pppp")
	s_load_b64 s[6:7], s[0:1], 0x20
	s_wait_dscnt 0x0
	v_add_f32_e32 v12, v12, v15
	ds_bpermute_b32 v9, v9, v12
	s_wait_dscnt 0x0
	v_add_f32_e32 v9, v12, v9
	ds_bpermute_b32 v10, v10, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v9, v9, v10
	ds_bpermute_b32 v10, v11, v9
	s_wait_dscnt 0x0
	v_add_f32_e32 v9, v9, v10
	ds_bpermute_b32 v10, v13, v9
	v_or_b32_e32 v13, 1, v31
	s_and_saveexec_b32 s1, s2
	s_cbranch_execz .LBB10_93
; %bb.91:
	s_delay_alu instid0(VALU_DEP_1)
	v_cmp_gt_u32_e64 s0, s61, v13
	s_and_b32 exec_lo, exec_lo, s0
	s_cbranch_execz .LBB10_93
; %bb.92:
	s_wait_dscnt 0x0
	v_dual_add_f32 v9, v9, v10 :: v_dual_lshlrev_b32 v10, 2, v31
	ds_store_b32 v10, v9 offset:14340
.LBB10_93:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_dual_mov_b32 v10, 0 :: v_dual_lshlrev_b32 v9, 9, v34
	v_lshlrev_b32_e32 v11, 6, v33
	v_lshlrev_b32_e32 v12, 5, v35
	s_lshl_b32 s0, ttmp9, 7
	v_and_b32_e32 v15, 8, v0
	v_and_b32_e32 v16, 1, v0
	s_wait_kmcnt 0x0
	s_cmp_lg_u32 s9, 0
	v_or3_b32 v11, v9, v11, v12
	s_wait_alu depctr_sa_sdst(0)
	v_lshl_or_b32 v9, v32, 2, s0
	v_cmp_ne_u32_e64 s1, 0, v15
	v_cmp_eq_u32_e64 s4, 1, v16
	v_cmp_lt_u32_e64 s0, 15, v32
	s_barrier_wait -1
	global_inv scope:SCOPE_SE
	ds_store_b128 v11, v[1:4]
	ds_store_b128 v11, v[5:8] offset:16
	s_wait_loadcnt_dscnt 0x0
	s_barrier_signal -1
	v_and_b32_e32 v2, 2, v0
	v_and_b32_e32 v3, 4, v0
	v_lshlrev_b64_e32 v[0:1], 1, v[9:10]
	v_lshlrev_b32_e32 v8, 4, v32
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
	s_cbranch_execz .LBB10_117
; %bb.94:
	v_lshlrev_b32_e32 v0, 2, v31
	v_lshl_or_b32 v1, v31, 9, v8
	v_xor_b32_e32 v7, 1, v14
	ds_load_b32 v6, v0 offset:14336
	ds_load_b128 v[0:3], v1
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	s_wait_dscnt 0x0
	v_dual_fmamk_f32 v0, v6, 0xc4bfc896, v0 :: v_dual_fmac_f32 v3, 0xc4bfc896, v6
	v_dual_fmamk_f32 v1, v6, 0xc4bfc896, v1 :: v_dual_fmamk_f32 v2, v6, 0xc4bfc896, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_cndmask_b32 v6, v14, v7 :: v_dual_sub_f32 v11, v0, v1
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
	ds_bpermute_b32 v15, v6, v0
	ds_bpermute_b32 v16, v6, v1
	ds_bpermute_b32 v17, v6, v2
	ds_bpermute_b32 v18, v6, v3
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s10, s4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execz .LBB10_96
; %bb.95:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v11, v15, v0 :: v_dual_sub_f32 v12, v16, v1
	s_wait_dscnt 0x0
	v_dual_sub_f32 v6, v17, v2 :: v_dual_sub_f32 v7, v18, v3
                                        ; implicit-def: $vgpr0
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr2
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr3
                                        ; implicit-def: $vgpr18
.LBB10_96:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB10_98
; %bb.97:
	s_wait_dscnt 0x2
	v_dual_add_f32 v11, v0, v15 :: v_dual_add_f32 v12, v1, v16
	s_wait_dscnt 0x0
	v_dual_add_f32 v6, v2, v17 :: v_dual_add_f32 v7, v3, v18
.LBB10_98:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_xor_b32_e32 v0, 2, v14
                                        ; implicit-def: $vgpr2_vgpr3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v14, v0, vcc_lo
	v_lshlrev_b32_e32 v0, 2, v0
	s_wait_dscnt 0x1
	ds_bpermute_b32 v17, v0, v11
	s_wait_dscnt 0x1
	ds_bpermute_b32 v18, v0, v12
	ds_bpermute_b32 v15, v0, v6
	ds_bpermute_b32 v16, v0, v7
                                        ; implicit-def: $vgpr0_vgpr1
	s_and_saveexec_b32 s10, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execz .LBB10_100
; %bb.99:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v1, v18, v12 :: v_dual_sub_f32 v0, v17, v11
	s_wait_dscnt 0x0
	v_dual_sub_f32 v3, v16, v7 :: v_dual_sub_f32 v2, v15, v6
                                        ; implicit-def: $vgpr12
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr11
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr15
.LBB10_100:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB10_102
; %bb.101:
	s_wait_dscnt 0x2
	v_dual_add_f32 v1, v12, v18 :: v_dual_add_f32 v0, v11, v17
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v7, v16 :: v_dual_add_f32 v2, v6, v15
.LBB10_102:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_xor_b32_e32 v6, 4, v14
                                        ; implicit-def: $vgpr11_vgpr12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v14, v6, vcc_lo
	v_lshlrev_b32_e32 v6, 2, v6
	s_wait_dscnt 0x3
	ds_bpermute_b32 v17, v6, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v18, v6, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v15, v6, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v6, v3
                                        ; implicit-def: $vgpr6_vgpr7
	s_and_saveexec_b32 s10, s2
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execz .LBB10_104
; %bb.103:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v7, v18, v1 :: v_dual_sub_f32 v6, v17, v0
	s_wait_dscnt 0x0
	v_dual_sub_f32 v12, v16, v3 :: v_dual_sub_f32 v11, v15, v2
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr15
.LBB10_104:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB10_106
; %bb.105:
	s_wait_dscnt 0x2
	v_dual_add_f32 v7, v1, v18 :: v_dual_add_f32 v6, v0, v17
	s_wait_dscnt 0x0
	v_dual_add_f32 v12, v3, v16 :: v_dual_add_f32 v11, v2, v15
.LBB10_106:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_xor_b32_e32 v0, 8, v14
                                        ; implicit-def: $vgpr2_vgpr3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v14, v0, vcc_lo
	v_lshlrev_b32_e32 v0, 2, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v17, v0, v6
	s_wait_dscnt 0x3
	ds_bpermute_b32 v18, v0, v7
	s_wait_dscnt 0x3
	ds_bpermute_b32 v15, v0, v11
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v0, v12
                                        ; implicit-def: $vgpr0_vgpr1
	s_and_saveexec_b32 s10, s1
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execz .LBB10_108
; %bb.107:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v1, v18, v7 :: v_dual_sub_f32 v0, v17, v6
	s_wait_dscnt 0x0
	v_dual_sub_f32 v3, v16, v12 :: v_dual_sub_f32 v2, v15, v11
                                        ; implicit-def: $vgpr6_vgpr7
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr11_vgpr12
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr15
.LBB10_108:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB10_110
; %bb.109:
	s_wait_dscnt 0x2
	v_dual_add_f32 v1, v7, v18 :: v_dual_add_f32 v0, v6, v17
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v12, v16 :: v_dual_add_f32 v2, v11, v15
.LBB10_110:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	v_xor_b32_e32 v6, 16, v14
                                        ; implicit-def: $vgpr11_vgpr12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v14, v6, vcc_lo
	v_lshlrev_b32_e32 v6, 2, v6
	s_wait_dscnt 0x3
	ds_bpermute_b32 v17, v6, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v18, v6, v1
	s_wait_dscnt 0x3
	ds_bpermute_b32 v15, v6, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v6, v3
                                        ; implicit-def: $vgpr6_vgpr7
	s_and_saveexec_b32 s10, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s10, exec_lo, s10
	s_cbranch_execz .LBB10_112
; %bb.111:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v7, v18, v1 :: v_dual_sub_f32 v6, v17, v0
	s_wait_dscnt 0x0
	v_dual_sub_f32 v12, v16, v3 :: v_dual_sub_f32 v11, v15, v2
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr15
.LBB10_112:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s10, s10
	s_cbranch_execz .LBB10_114
; %bb.113:
	s_wait_dscnt 0x2
	v_dual_add_f32 v7, v1, v18 :: v_dual_add_f32 v6, v0, v17
	s_wait_dscnt 0x0
	v_dual_add_f32 v12, v3, v16 :: v_dual_add_f32 v11, v2, v15
.LBB10_114:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s10
	s_wait_dscnt 0x0
	global_load_b64 v[15:16], v[4:5], off
	v_dual_mul_f32 v3, 0x3a1ccd09, v12 :: v_dual_add_nc_u32 v0, s60, v31
	v_dual_mul_f32 v1, 0x3a1ccd09, v7 :: v_dual_mul_f32 v2, 0x3a1ccd09, v6
	v_mul_f32_e32 v11, 0x3a1ccd09, v11
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[6:7], null, v0, s8, v[9:10]
	s_and_b32 vcc_lo, exec_lo, s5
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v1, v15, v1, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v0, v15, v2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v3, v16, v3, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v2, v16, v11, neg(0) op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccz .LBB10_145
; %bb.115:
	v_lshlrev_b64_e32 v[11:12], 2, v[6:7]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v11, vcc_lo, s6, v11
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v12, null, s7, v12, vcc_lo
	global_store_b128 v[11:12], v[0:3], off
	s_cbranch_execnz .LBB10_117
.LBB10_116:
	v_lshlrev_b64_e32 v[6:7], 1, v[6:7]
	v_bfe_u32 v12, v0, 16, 1
	v_bfe_u32 v15, v2, 16, 1
	v_or_b32_e32 v17, 0x400000, v0
	v_bfe_u32 v11, v1, 16, 1
	v_or_b32_e32 v19, 0x400000, v2
	v_add_co_u32 v6, vcc_lo, s6, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s7, v7, vcc_lo
	v_add3_u32 v12, v12, v0, 0x7fff
	v_cmp_u_f32_e32 vcc_lo, v0, v0
	v_add3_u32 v15, v15, v2, 0x7fff
	v_add3_u32 v11, v11, v1, 0x7fff
	v_or_b32_e32 v16, 0x400000, v1
	v_bfe_u32 v18, v3, 16, 1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v12, v12, v17, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v2, v2
	s_delay_alu instid0(VALU_DEP_3)
	v_add3_u32 v17, v18, v3, 0x7fff
	v_or_b32_e32 v18, 0x400000, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v2, v15, v19, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v1, v1
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v11, v16, vcc_lo
	v_cmp_u_f32_e32 vcc_lo, v3, v3
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v1, v17, v18, vcc_lo
	v_mov_b16_e32 v1.l, v2.h
	v_mov_b16_e32 v0.l, v12.h
	global_store_b64 v[6:7], v[0:1], off
.LBB10_117:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s9
	s_delay_alu instid0(SALU_CYCLE_1)
	s_mov_b32 s9, exec_lo
	v_cmpx_gt_u32_e64 s61, v13
	s_cbranch_execz .LBB10_141
; %bb.118:
	v_lshlrev_b32_e32 v0, 2, v31
	v_lshl_or_b32 v1, v13, 9, v8
	v_xor_b32_e32 v7, 1, v14
	ds_load_b32 v6, v0 offset:14340
	ds_load_b128 v[0:3], v1
	v_cmp_gt_u32_e32 vcc_lo, 32, v7
	s_wait_dscnt 0x0
	v_dual_fmamk_f32 v0, v6, 0xc4bfc896, v0 :: v_dual_fmac_f32 v3, 0xc4bfc896, v6
	v_dual_fmamk_f32 v1, v6, 0xc4bfc896, v1 :: v_dual_fmamk_f32 v2, v6, 0xc4bfc896, v2
	s_wait_alu depctr_va_vcc(0)
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_2)
	v_dual_cndmask_b32 v6, v14, v7 :: v_dual_add_f32 v7, v0, v1
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
	ds_bpermute_b32 v15, v6, v1
	ds_bpermute_b32 v16, v6, v2
	ds_bpermute_b32 v17, v6, v3
                                        ; implicit-def: $vgpr6
	s_and_saveexec_b32 s9, s4
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s4, exec_lo, s9
	s_cbranch_execz .LBB10_120
; %bb.119:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v8, v12, v0 :: v_dual_sub_f32 v11, v15, v1
	s_wait_dscnt 0x0
	v_dual_sub_f32 v6, v16, v2 :: v_dual_sub_f32 v7, v17, v3
                                        ; implicit-def: $vgpr0
                                        ; implicit-def: $vgpr12
                                        ; implicit-def: $vgpr1
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr2
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr3
                                        ; implicit-def: $vgpr17
.LBB10_120:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s4, s4
	s_cbranch_execz .LBB10_122
; %bb.121:
	s_wait_dscnt 0x2
	v_dual_add_f32 v8, v0, v12 :: v_dual_add_f32 v11, v1, v15
	s_wait_dscnt 0x0
	v_dual_add_f32 v6, v2, v16 :: v_dual_add_f32 v7, v3, v17
.LBB10_122:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s4
	v_xor_b32_e32 v0, 2, v14
                                        ; implicit-def: $vgpr2_vgpr3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v14, v0, vcc_lo
	v_lshlrev_b32_e32 v0, 2, v0
	s_wait_dscnt 0x1
	ds_bpermute_b32 v16, v0, v8
	s_wait_dscnt 0x1
	ds_bpermute_b32 v17, v0, v11
	ds_bpermute_b32 v12, v0, v6
	ds_bpermute_b32 v15, v0, v7
                                        ; implicit-def: $vgpr0_vgpr1
	s_and_saveexec_b32 s4, s3
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s3, exec_lo, s4
	s_cbranch_execz .LBB10_124
; %bb.123:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v1, v17, v11 :: v_dual_sub_f32 v0, v16, v8
	s_wait_dscnt 0x0
	v_dual_sub_f32 v3, v15, v7 :: v_dual_sub_f32 v2, v12, v6
                                        ; implicit-def: $vgpr11
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr7
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr6
                                        ; implicit-def: $vgpr12
.LBB10_124:
	s_and_not1_saveexec_b32 s3, s3
	s_cbranch_execz .LBB10_126
; %bb.125:
	s_wait_dscnt 0x2
	v_dual_add_f32 v1, v11, v17 :: v_dual_add_f32 v0, v8, v16
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v7, v15 :: v_dual_add_f32 v2, v6, v12
.LBB10_126:
	s_or_b32 exec_lo, exec_lo, s3
	v_xor_b32_e32 v6, 4, v14
                                        ; implicit-def: $vgpr11_vgpr12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v14, v6, vcc_lo
	v_lshlrev_b32_e32 v6, 2, v6
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v6, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v17, v6, v1
	ds_bpermute_b32 v8, v6, v2
	s_wait_dscnt 0x3
	ds_bpermute_b32 v15, v6, v3
                                        ; implicit-def: $vgpr6_vgpr7
	s_and_saveexec_b32 s3, s2
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s2, exec_lo, s3
	s_cbranch_execz .LBB10_128
; %bb.127:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v7, v17, v1 :: v_dual_sub_f32 v6, v16, v0
	s_wait_dscnt 0x0
	v_dual_sub_f32 v12, v15, v3 :: v_dual_sub_f32 v11, v8, v2
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr8
.LBB10_128:
	s_and_not1_saveexec_b32 s2, s2
	s_cbranch_execz .LBB10_130
; %bb.129:
	s_wait_dscnt 0x2
	v_dual_add_f32 v7, v1, v17 :: v_dual_add_f32 v6, v0, v16
	s_wait_dscnt 0x0
	v_dual_add_f32 v12, v3, v15 :: v_dual_add_f32 v11, v2, v8
.LBB10_130:
	s_or_b32 exec_lo, exec_lo, s2
	v_xor_b32_e32 v0, 8, v14
                                        ; implicit-def: $vgpr2_vgpr3
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v0
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v0, v14, v0, vcc_lo
	v_lshlrev_b32_e32 v0, 2, v0
	s_wait_dscnt 0x3
	ds_bpermute_b32 v16, v0, v6
	s_wait_dscnt 0x3
	ds_bpermute_b32 v17, v0, v7
	s_wait_dscnt 0x3
	ds_bpermute_b32 v8, v0, v11
	s_wait_dscnt 0x3
	ds_bpermute_b32 v15, v0, v12
                                        ; implicit-def: $vgpr0_vgpr1
	s_and_saveexec_b32 s2, s1
	s_delay_alu instid0(SALU_CYCLE_1)
	s_xor_b32 s1, exec_lo, s2
	s_cbranch_execz .LBB10_132
; %bb.131:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v1, v17, v7 :: v_dual_sub_f32 v0, v16, v6
	s_wait_dscnt 0x0
	v_dual_sub_f32 v3, v15, v12 :: v_dual_sub_f32 v2, v8, v11
                                        ; implicit-def: $vgpr6_vgpr7
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr11_vgpr12
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr8
.LBB10_132:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s1, s1
	s_cbranch_execz .LBB10_134
; %bb.133:
	s_wait_dscnt 0x2
	v_dual_add_f32 v1, v7, v17 :: v_dual_add_f32 v0, v6, v16
	s_wait_dscnt 0x0
	v_dual_add_f32 v3, v12, v15 :: v_dual_add_f32 v2, v11, v8
.LBB10_134:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s1
	v_xor_b32_e32 v6, 16, v14
                                        ; implicit-def: $vgpr11_vgpr12
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)
	v_cmp_gt_u32_e32 vcc_lo, 32, v6
	s_wait_alu depctr_va_vcc(0)
	v_cndmask_b32_e32 v6, v14, v6, vcc_lo
	v_lshlrev_b32_e32 v6, 2, v6
	s_wait_dscnt 0x0
	ds_bpermute_b32 v15, v6, v0
	ds_bpermute_b32 v16, v6, v1
	ds_bpermute_b32 v8, v6, v2
	ds_bpermute_b32 v14, v6, v3
                                        ; implicit-def: $vgpr6_vgpr7
	s_and_saveexec_b32 s1, s0
	s_wait_alu depctr_sa_sdst(0)
	s_xor_b32 s0, exec_lo, s1
	s_cbranch_execz .LBB10_136
; %bb.135:
	s_wait_dscnt 0x2
	v_dual_sub_f32 v7, v16, v1 :: v_dual_sub_f32 v6, v15, v0
	s_wait_dscnt 0x0
	v_dual_sub_f32 v12, v14, v3 :: v_dual_sub_f32 v11, v8, v2
                                        ; implicit-def: $vgpr0_vgpr1
                                        ; implicit-def: $vgpr16
                                        ; implicit-def: $vgpr15
                                        ; implicit-def: $vgpr2_vgpr3
                                        ; implicit-def: $vgpr14
                                        ; implicit-def: $vgpr8
.LBB10_136:
	s_wait_alu depctr_sa_sdst(0)
	s_and_not1_saveexec_b32 s0, s0
	s_cbranch_execz .LBB10_138
; %bb.137:
	s_wait_dscnt 0x2
	v_dual_add_f32 v7, v1, v16 :: v_dual_add_f32 v6, v0, v15
	s_wait_dscnt 0x0
	v_dual_add_f32 v12, v3, v14 :: v_dual_add_f32 v11, v2, v8
.LBB10_138:
	s_wait_alu depctr_sa_sdst(0)
	s_or_b32 exec_lo, exec_lo, s0
	s_wait_dscnt 0x0
	global_load_b64 v[14:15], v[4:5], off
	v_dual_mul_f32 v1, 0x3a1ccd09, v7 :: v_dual_add_nc_u32 v0, s60, v13
	v_dual_mul_f32 v2, 0x3a1ccd09, v6 :: v_dual_mul_f32 v3, 0x3a1ccd09, v12
	v_mul_f32_e32 v6, 0x3a1ccd09, v11
	s_delay_alu instid0(VALU_DEP_3)
	v_mad_co_u64_u32 v[4:5], null, v0, s8, v[9:10]
	s_and_not1_b32 vcc_lo, exec_lo, s5
	s_wait_loadcnt 0x0
	v_fma_mix_f32 v1, v14, v1, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v0, v14, v2, neg(0) op_sel_hi:[1,0,0]
	v_fma_mix_f32 v3, v15, v3, neg(0) op_sel:[1,0,0] op_sel_hi:[1,0,0]
	v_fma_mix_f32 v2, v15, v6, neg(0) op_sel_hi:[1,0,0]
	s_wait_alu depctr_sa_sdst(0)
	s_cbranch_vccnz .LBB10_146
; %bb.139:
	v_lshlrev_b64_e32 v[6:7], 2, v[4:5]
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_2)
	v_add_co_u32 v6, vcc_lo, s6, v6
	s_wait_alu depctr_va_vcc(0)
	v_add_co_ci_u32_e64 v7, null, s7, v7, vcc_lo
	global_store_b128 v[6:7], v[0:3], off
	s_cbranch_execnz .LBB10_141
.LBB10_140:
	v_lshlrev_b64_e32 v[4:5], 1, v[4:5]
	v_bfe_u32 v7, v0, 16, 1
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
.LBB10_141:
	s_endpgm
.LBB10_142:
	s_mov_b32 s2, 0
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr12
                                        ; implicit-def: $vgpr31
	s_cbranch_execnz .LBB10_86
	s_branch .LBB10_87
.LBB10_143:
	s_mov_b32 s2, 0
                                        ; implicit-def: $vgpr12
.LBB10_144:
	s_trap 2
	; divergent unreachable
	s_branch .LBB10_87
.LBB10_145:
	s_branch .LBB10_116
.LBB10_146:
	s_branch .LBB10_140
.Lfunc_end10:
