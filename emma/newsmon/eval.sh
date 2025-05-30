./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s1710_lr3e-05 -c newsmon_sl_p1_s0
./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s2573_lr3e-05 -c newsmon_sl_p1_s0
./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s3821_lr3e-05 -c newsmon_sl_p1_s0
./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s4823_lr3e-05 -c newsmon_sl_p1_s0
./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s7352_lr3e-05 -c newsmon_sl_p1_s0

./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s1710_lr3e-05 -c newsmon_sl_p1_s0 --test_l_class Rare
./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s2573_lr3e-05 -c newsmon_sl_p1_s0 --test_l_class Rare
./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s3821_lr3e-05 -c newsmon_sl_p1_s0 --test_l_class Rare
./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s4823_lr3e-05 -c newsmon_sl_p1_s0 --test_l_class Rare
./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s7352_lr3e-05 -c newsmon_sl_p1_s0 --test_l_class Rare

./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s1710_lr3e-05 -c newsmon_sl_p1_s0 --test_l_class Frequent
./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s2573_lr3e-05 -c newsmon_sl_p1_s0 --test_l_class Frequent
./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s3821_lr3e-05 -c newsmon_sl_p1_s0 --test_l_class Frequent
./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s4823_lr3e-05 -c newsmon_sl_p1_s0 --test_l_class Frequent
./newsmon te test --ptm_name xlmrb_newsmon_sl_p1_s0_x4_b16_e30_s7352_lr3e-05 -c newsmon_sl_p1_s0 --test_l_class Frequent

./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s2611_lr3e-05 -c eurlex_all_p0_s0
./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s2963_lr3e-05 -c eurlex_all_p0_s0
./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s4789_lr3e-05 -c eurlex_all_p0_s0
./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s5823_lr3e-05 -c eurlex_all_p0_s0
./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s7681_lr3e-05 -c eurlex_all_p0_s0

./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s2611_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Rare
./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s2963_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Rare
./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s4789_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Rare
./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s5823_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Rare
./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s7681_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Rare

./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s2611_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Frequent
./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s2963_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Frequent
./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s4789_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Frequent
./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s5823_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Frequent
./newsmon te test --ptm_name xlmrb_eurlex_x0_b16_e30_s7681_lr3e-05 -c eurlex_all_p0_s0 --test_l_class Frequent

./newsmon fa test_zshot -c newsmon -l sl --public --ptm_models bge_m3,jina3,gte,n_bge_m3
./newsmon fa test_zshot -c newsmon -l sl --public --ptm_models bge_m3,jina3,gte,n_bge_m3 --test_l_class Rare
./newsmon fa test_zshot -c newsmon -l sl --public --ptm_models bge_m3,jina3,gte,n_bge_m3 --test_l_class Frequent
./newsmon fa test_zshot -c eurlex --ptm_models bge_m3,jina3,gte,e_bge_m3
./newsmon fa test_zshot -c eurlex --ptm_models bge_m3,jina3,gte,e_bge_m3 --test_l_class Rare
./newsmon fa test_zshot -c eurlex --ptm_models bge_m3,jina3,gte,e_bge_m3 --test_l_class Frequent

./newsmon fa test_mlknn -c newsmon -l sl --public --ptm_models bge_m3
#./newsmon fa test_mlknn -c newsmon -l sl --public --ptm_models jina3
./newsmon fa test_mlknn -c newsmon -l sl --public --ptm_models gte
./newsmon fa test_mlknn -c newsmon -l sl --public --ptm_models n_bge_m3
./newsmon fa test_mlknn -c newsmon -l sl --public --ptm_models bge_m3 --test_l_class Rare
#./newsmon fa test_mlknn -c newsmon -l sl --public --ptm_models jina3 --test_l_class Rare
./newsmon fa test_mlknn -c newsmon -l sl --public --ptm_models gte --test_l_class Rare
./newsmon fa test_mlknn -c newsmon -l sl --public --ptm_models n_bge_m3 --test_l_class Rare
./newsmon fa test_mlknn -c newsmon -l sl --public --ptm_models bge_m3 --test_l_class Frequent
#./newsmon fa test_mlknn -c newsmon -l sl --public --ptm_models jina3 --test_l_class Frequent
./newsmon fa test_mlknn -c newsmon -l sl --public --ptm_models gte --test_l_class Frequent
./newsmon fa test_mlknn -c newsmon -l sl --public --ptm_models n_bge_m3 --test_l_class Frequent

./newsmon fa test_mlknn -c eurlex --ptm_models bge_m3
#./newsmon fa test_mlknn -c eurlex --ptm_models jina3
./newsmon fa test_mlknn -c eurlex --ptm_models gte
./newsmon fa test_mlknn -c eurlex --ptm_models e_bge_m3
./newsmon fa test_mlknn -c eurlex --ptm_models bge_m3 --test_l_class Rare
#./newsmon fa test_mlknn -c eurlex --ptm_models jina3 --test_l_class Rare
./newsmon fa test_mlknn -c eurlex --ptm_models gte --test_l_class Rare
./newsmon fa test_mlknn -c eurlex --ptm_models e_bge_m3 --test_l_class Rare
./newsmon fa test_mlknn -c eurlex --ptm_models bge_m3 --test_l_class Frequent
#./newsmon fa test_mlknn -c eurlex --ptm_models jina3 --test_l_class Frequent
./newsmon fa test_mlknn -c eurlex --ptm_models gte --test_l_class Frequent
./newsmon fa test_mlknn -c eurlex --ptm_models e_bge_m3 --test_l_class Frequent

./newsmon fa test_rae -c newsmon -l sl --public --ptm_models bge_m3,jina3,gte,n_bge_m3
./newsmon fa test_rae -c newsmon -l sl --public --ptm_models bge_m3,jina3,gte,n_bge_m3 --test_l_class Rare
./newsmon fa test_rae -c newsmon -l sl --public --ptm_models bge_m3,jina3,gte,n_bge_m3 --test_l_class Frequent
./newsmon fa test_rae -c eurlex --ptm_models bge_m3,jina3,gte,e_bge_m3
./newsmon fa test_rae -c eurlex --ptm_models bge_m3,jina3,gte,e_bge_m3 --test_l_class Rare
./newsmon fa test_rae -c eurlex --ptm_models bge_m3,jina3,gte,e_bge_m3 --test_l_class Frequent

./newsmon fa test_rae_sim -c newsmon -l sl --public --ptm_models bge_m3,jina3,gte,n_bge_m3
./newsmon fa test_rae_sim -c newsmon -l sl --public --ptm_models bge_m3,jina3,gte,n_bge_m3 --test_l_class Rare
./newsmon fa test_rae_sim -c newsmon -l sl --public --ptm_models bge_m3,jina3,gte,n_bge_m3 --test_l_class Frequent
./newsmon fa test_rae_sim -c eurlex --ptm_models bge_m3,jina3,gte,e_bge_m3
./newsmon fa test_rae_sim -c eurlex --ptm_models bge_m3,jina3,gte,e_bge_m3 --test_l_class Rare
./newsmon fa test_rae_sim -c eurlex --ptm_models bge_m3,jina3,gte,e_bge_m3 --test_l_class Frequent

# after hp search
./newsmon fa test_rae -c newsmon -l sl --public --ptm_models bge_m3,n_bge_m3
./newsmon fa test_rae -c newsmon -l sl --public --ptm_models bge_m3,n_bge_m3 --test_l_class Rare
./newsmon fa test_rae -c newsmon -l sl --public --ptm_models bge_m3,n_bge_m3 --test_l_class Frequent
./newsmon fa test_rae -c eurlex --ptm_models bge_m3,e_bge_m3
./newsmon fa test_rae -c eurlex --ptm_models bge_m3,e_bge_m3 --test_l_class Rare
./newsmon fa test_rae -c eurlex --ptm_models bge_m3,e_bge_m3 --test_l_class Frequent

./newsmon fa test_rae_sim -c newsmon -l sl --public --ptm_models bge_m3,n_bge_m3
./newsmon fa test_rae_sim -c newsmon -l sl --public --ptm_models bge_m3,n_bge_m3 --test_l_class Rare
./newsmon fa test_rae_sim -c newsmon -l sl --public --ptm_models bge_m3,n_bge_m3 --test_l_class Frequent
./newsmon fa test_rae_sim -c eurlex --ptm_models bge_m3,e_bge_m3
./newsmon fa test_rae_sim -c eurlex --ptm_models bge_m3,e_bge_m3 --test_l_class Rare
./newsmon fa test_rae_sim -c eurlex --ptm_models bge_m3,e_bge_m3 --test_l_class Frequent

./newsmon bl majority -c newsmon -l sl --public
./newsmon bl majority -c newsmon -l sl --public --test_l_class Rare
./newsmon bl majority -c newsmon -l sl --public --test_l_class Frequent
./newsmon bl majority -c eurlex
./newsmon bl majority -c eurlex --test_l_class Rare
./newsmon bl majority -c eurlex --test_l_class Frequent

./newsmon bl random -c newsmon -l sl --public
./newsmon bl random -c newsmon -l sl --public --test_l_class Rare
./newsmon bl random -c newsmon -l sl --public --test_l_class Frequent
./newsmon bl random -c eurlex
./newsmon bl random -c eurlex --test_l_class Rare
./newsmon bl random -c eurlex --test_l_class Frequent

./newsmon bl svm -c newsmon -l sl --public
./newsmon bl svm -c newsmon -l sl --public --test_l_class Rare
./newsmon bl svm -c newsmon -l sl --public --test_l_class Frequent
./newsmon bl svm -c eurlex
./newsmon bl svm -c eurlex --test_l_class Rare
./newsmon bl svm -c eurlex --test_l_class Frequent

./newsmon bl logreg -c newsmon -l sl --public
./newsmon bl logreg -c newsmon -l sl --public --test_l_class Rare
./newsmon bl logreg -c newsmon -l sl --public --test_l_class Frequent
./newsmon bl logreg -c eurlex
./newsmon bl logreg -c eurlex --test_l_class Rare
./newsmon bl logreg -c eurlex --test_l_class Frequent
