./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s1710_lr3e-05 -c mulabel_sl_p1_s0
./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s2573_lr3e-05 -c mulabel_sl_p1_s0
./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s3821_lr3e-05 -c mulabel_sl_p1_s0
./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s4823_lr3e-05 -c mulabel_sl_p1_s0
./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s7352_lr3e-05 -c mulabel_sl_p1_s0

./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s1710_lr3e-05 -c mulabel_sl_p1_s0 --test_l_class Rare
./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s2573_lr3e-05 -c mulabel_sl_p1_s0 --test_l_class Rare
./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s3821_lr3e-05 -c mulabel_sl_p1_s0 --test_l_class Rare
./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s4823_lr3e-05 -c mulabel_sl_p1_s0 --test_l_class Rare
./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s7352_lr3e-05 -c mulabel_sl_p1_s0 --test_l_class Rare

./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s1710_lr3e-05 -c mulabel_sl_p1_s0 --test_l_class Frequent
./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s2573_lr3e-05 -c mulabel_sl_p1_s0 --test_l_class Frequent
./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s3821_lr3e-05 -c mulabel_sl_p1_s0 --test_l_class Frequent
./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s4823_lr3e-05 -c mulabel_sl_p1_s0 --test_l_class Frequent
./mulabel te test --ptm_name xlmrb_mulabel_sl_p1_s0_x4_b16_e30_s7352_lr3e-05 -c mulabel_sl_p1_s0 --test_l_class Frequent

./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s2611_lr3e-05 -c eurlex_all_p0_s0
./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s2963_lr3e-05 -c eurlex_all_p0_s0
./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s4789_lr3e-05 -c eurlex_all_p0_s0
./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s5823_lr3e-05 -c eurlex_all_p0_s0
./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s7681_lr3e-05 -c eurlex_all_p0_s0

./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s2611_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Rare
./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s2963_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Rare
./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s4789_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Rare
./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s5823_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Rare
./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s7681_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Rare

./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s2611_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Frequent
./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s2963_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Frequent
./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s4789_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Frequent
./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s5823_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Frequent
./mulabel te test --ptm_name xlmrb_eurlex_x0_b16_e30_s7681_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Frequent

./mulabel fa test_rae -c mulabel -l sl --public --ptm_models bge_m3,jinav3,gte && \                                                                                                                                                                                                                25-01-16T15:11:11 ⎇ main*
./mulabel fa test_rae -c mulabel -l sl --public --ptm_models bge_m3,jinav3,gte --test_l_class Rare && \
./mulabel fa test_rae -c mulabel -l sl --public --ptm_models bge_m3,jinav3,gte --test_l_class Frequent && \
./mulabel fa test_rae -c eurlex --ptm_models bge_m3,jinav3,gte && \
./mulabel fa test_rae -c eurlex --ptm_models bge_m3,jinav3,gte --test_l_class Rare && \
./mulabel fa test_rae -c eurlex --ptm_models bge_m3,jinav3,gte --test_l_class Frequent
