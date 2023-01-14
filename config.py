import pathlib

from utils.langs import lang_list

# get base dir
base_dir = pathlib.Path(__file__).parent.resolve()

# all the language you want to translate when calling translate_all_langs in translator.py, you can add up to 200 langs, check
# utils.langs to know the codification for all the languages
translation_langs = ["spa_Latn", "eng_Latn", "fra_Latn", "zho_Hans"]
# translation_langs = lang_list

# full path of LID model
rel_path = "weights/lid218e.bin"
lid_path = base_dir.joinpath(rel_path).as_posix().replace('\\', '/')

# full path of toxicity vocabulary
tox_folder = "data"
path_toxicity_data = base_dir.joinpath(tox_folder).as_posix().replace('\\', '/')

toxicity_files = {
    "ace_Arab": "{}/ace_Arab_twl.txt".format(path_toxicity_data),
    "ace_Latn": "{}/ace_Latn_twl.txt".format(path_toxicity_data),
    "acm_Arab": "{}/acm_Arab_twl.txt".format(path_toxicity_data),
    "acq_Arab": "{}/acq_Arab_twl.txt".format(path_toxicity_data),
    "aeb_Arab": "{}/aeb_Arab_twl.txt".format(path_toxicity_data),
    "afr_Latn": "{}/afr_Latn_twl.txt".format(path_toxicity_data),
    "ajp_Arab": "{}/ajp_Arab_twl.txt".format(path_toxicity_data),
    "aka_Latn": "{}/aka_Latn_twl.txt".format(path_toxicity_data),
    "als_Latn": "{}/als_Latn_twl.txt".format(path_toxicity_data),
    "amh_Ethi": "{}/amh_Ethi_twl.txt".format(path_toxicity_data),
    "apc_Arab": "{}/apc_Arab_twl.txt".format(path_toxicity_data),
    "arb_Arab": "{}/arb_Arab_twl.txt".format(path_toxicity_data),
    "arb_Latn": "{}/arb_Latn_twl.txt".format(path_toxicity_data),
    "ars_Arab": "{}/ars_Arab_twl.txt".format(path_toxicity_data),
    "ary_Arab": "{}/ary_Arab_twl.txt".format(path_toxicity_data),
    "arz_Arab": "{}/arz_Arab_twl.txt".format(path_toxicity_data),
    "asm_Beng": "{}/asm_Beng_twl.txt".format(path_toxicity_data),
    "ast_Latn": "{}/ast_Latn_twl.txt".format(path_toxicity_data),
    "awa_Deva": "{}/awa_Deva_twl.txt".format(path_toxicity_data),
    "ayr_Latn": "{}/ayr_Latn_twl.txt".format(path_toxicity_data),
    "azb_Arab": "{}/azb_Arab_twl.txt".format(path_toxicity_data),
    "azj_Latn": "{}/azj_Latn_twl.txt".format(path_toxicity_data),
    "bak_Cyrl": "{}/bak_Cyrl_twl.txt".format(path_toxicity_data),
    "bam_Latn": "{}/bam_Latn_twl.txt".format(path_toxicity_data),
    "ban_Latn": "{}/ban_Latn_twl.txt".format(path_toxicity_data),
    "bel_Cyrl": "{}/bel_Cyrl_twl.txt".format(path_toxicity_data),
    "bem_Latn": "{}/bem_Latn_twl.txt".format(path_toxicity_data),
    "ben_Beng": "{}/ben_Beng_twl.txt".format(path_toxicity_data),
    "bho_Deva": "{}/bho_Deva_twl.txt".format(path_toxicity_data),
    "bjn_Arab": "{}/bjn_Arab_twl.txt".format(path_toxicity_data),
    "bjn_Latn": "{}/bjn_Latn_twl.txt".format(path_toxicity_data),
    "bod_Tibt": "{}/bod_Tibt_twl.txt".format(path_toxicity_data),
    "bos_Latn": "{}/bos_Latn_twl.txt".format(path_toxicity_data),
    "bug_Latn": "{}/bug_Latn_twl.txt".format(path_toxicity_data),
    "bul_Cyrl": "{}/bul_Cyrl_twl.txt".format(path_toxicity_data),
    "cat_Latn": "{}/cat_Latn_twl.txt".format(path_toxicity_data),
    "ceb_Latn": "{}/ceb_Latn_twl.txt".format(path_toxicity_data),
    "ces_Latn": "{}/ces_Latn_twl.txt".format(path_toxicity_data),
    "cjk_Latn": "{}/cjk_Latn_twl.txt".format(path_toxicity_data),
    "ckb_Arab": "{}/ckb_Arab_twl.txt".format(path_toxicity_data),
    "crh_Latn": "{}/crh_Latn_twl.txt".format(path_toxicity_data),
    "cym_Latn": "{}/cym_Latn_twl.txt".format(path_toxicity_data),
    "dan_Latn": "{}/dan_Latn_twl.txt".format(path_toxicity_data),
    "deu_Latn": "{}/deu_Latn_twl.txt".format(path_toxicity_data),
    "dik_Latn": "{}/dik_Latn_twl.txt".format(path_toxicity_data),
    "dyu_Latn": "{}/dyu_Latn_twl.txt".format(path_toxicity_data),
    "dzo_Tibt": "{}/dzo_Tibt_twl.txt".format(path_toxicity_data),
    "ell_Grek": "{}/ell_Grek_twl.txt".format(path_toxicity_data),
    "eng_Latn": "{}/eng_Latn_twl.txt".format(path_toxicity_data),
    "epo_Latn": "{}/epo_Latn_twl.txt".format(path_toxicity_data),
    "est_Latn": "{}/est_Latn_twl.txt".format(path_toxicity_data),
    "eus_Latn": "{}/eus_Latn_twl.txt".format(path_toxicity_data),
    "ewe_Latn": "{}/ewe_Latn_twl.txt".format(path_toxicity_data),
    "fao_Latn": "{}/fao_Latn_twl.txt".format(path_toxicity_data),
    "fij_Latn": "{}/fij_Latn_twl.txt".format(path_toxicity_data),
    "fin_Latn": "{}/fin_Latn_twl.txt".format(path_toxicity_data),
    "fon_Latn": "{}/fon_Latn_twl.txt".format(path_toxicity_data),
    "fra_Latn": "{}/fra_Latn_twl.txt".format(path_toxicity_data),
    "fur_Latn": "{}/fur_Latn_twl.txt".format(path_toxicity_data),
    "fuv_Latn": "{}/fuv_Latn_twl.txt".format(path_toxicity_data),
    "gaz_Latn": "{}/gaz_Latn_twl.txt".format(path_toxicity_data),
    "gla_Latn": "{}/gla_Latn_twl.txt".format(path_toxicity_data),
    "gle_Latn": "{}/gle_Latn_twl.txt".format(path_toxicity_data),
    "glg_Latn": "{}/glg_Latn_twl.txt".format(path_toxicity_data),
    "grn_Latn": "{}/grn_Latn_twl.txt".format(path_toxicity_data),
    "guj_Gujr": "{}/guj_Gujr_twl.txt".format(path_toxicity_data),
    "hat_Latn": "{}/hat_Latn_twl.txt".format(path_toxicity_data),
    "hau_Latn": "{}/hau_Latn_twl.txt".format(path_toxicity_data),
    "heb_Hebr": "{}/heb_Hebr_twl.txt".format(path_toxicity_data),
    "hin_Deva": "{}/hin_Deva_twl.txt".format(path_toxicity_data),
    "hne_Deva": "{}/hne_Deva_twl.txt".format(path_toxicity_data),
    "hrv_Latn": "{}/hrv_Latn_twl.txt".format(path_toxicity_data),
    "hun_Latn": "{}/hun_Latn_twl.txt".format(path_toxicity_data),
    "hye_Armn": "{}/hye_Armn_twl.txt".format(path_toxicity_data),
    "ibo_Latn": "{}/ibo_Latn_twl.txt".format(path_toxicity_data),
    "ilo_Latn": "{}/ilo_Latn_twl.txt".format(path_toxicity_data),
    "ind_Latn": "{}/ind_Latn_twl.txt".format(path_toxicity_data),
    "isl_Latn": "{}/isl_Latn_twl.txt".format(path_toxicity_data),
    "ita_Latn": "{}/ita_Latn_twl.txt".format(path_toxicity_data),
    "jav_Latn": "{}/jav_Latn_twl.txt".format(path_toxicity_data),
    "jpn_Jpan": "{}/jpn_Jpan_twl.txt".format(path_toxicity_data),
    "kab_Latn": "{}/kab_Latn_twl.txt".format(path_toxicity_data),
    "kac_Latn": "{}/kac_Latn_twl.txt".format(path_toxicity_data),
    "kam_Latn": "{}/kam_Latn_twl.txt".format(path_toxicity_data),
    "kan_Knda": "{}/kan_Knda_twl.txt".format(path_toxicity_data),
    "kas_Arab": "{}/kas_Arab_twl.txt".format(path_toxicity_data),
    "kas_Deva": "{}/kas_Deva_twl.txt".format(path_toxicity_data),
    "kat_Geor": "{}/kat_Geor_twl.txt".format(path_toxicity_data),
    "kaz_Cyrl": "{}/kaz_Cyrl_twl.txt".format(path_toxicity_data),
    "kbp_Latn": "{}/kbp_Latn_twl.txt".format(path_toxicity_data),
    "kea_Latn": "{}/kea_Latn_twl.txt".format(path_toxicity_data),
    "khk_Cyrl": "{}/khk_Cyrl_twl.txt".format(path_toxicity_data),
    "khm_Khmr": "{}/khm_Khmr_twl.txt".format(path_toxicity_data),
    "kik_Latn": "{}/kik_Latn_twl.txt".format(path_toxicity_data),
    "kin_Latn": "{}/kin_Latn_twl.txt".format(path_toxicity_data),
    "kir_Cyrl": "{}/kir_Cyrl_twl.txt".format(path_toxicity_data),
    "kmb_Latn": "{}/kmb_Latn_twl.txt".format(path_toxicity_data),
    "kmr_Latn": "{}/kmr_Latn_twl.txt".format(path_toxicity_data),
    "knc_Arab": "{}/knc_Arab_twl.txt".format(path_toxicity_data),
    "knc_Latn": "{}/knc_Latn_twl.txt".format(path_toxicity_data),
    "kon_Latn": "{}/kon_Latn_twl.txt".format(path_toxicity_data),
    "kor_Hang": "{}/kor_Hang_twl.txt".format(path_toxicity_data),
    "lao_Laoo": "{}/lao_Laoo_twl.txt".format(path_toxicity_data),
    "lij_Latn": "{}/lij_Latn_twl.txt".format(path_toxicity_data),
    "lim_Latn": "{}/lim_Latn_twl.txt".format(path_toxicity_data),
    "lin_Latn": "{}/lin_Latn_twl.txt".format(path_toxicity_data),
    "lit_Latn": "{}/lit_Latn_twl.txt".format(path_toxicity_data),
    "lmo_Latn": "{}/lmo_Latn_twl.txt".format(path_toxicity_data),
    "ltg_Latn": "{}/ltg_Latn_twl.txt".format(path_toxicity_data),
    "ltz_Latn": "{}/ltz_Latn_twl.txt".format(path_toxicity_data),
    "lua_Latn": "{}/lua_Latn_twl.txt".format(path_toxicity_data),
    "lug_Latn": "{}/lug_Latn_twl.txt".format(path_toxicity_data),
    "luo_Latn": "{}/luo_Latn_twl.txt".format(path_toxicity_data),
    "lus_Latn": "{}/lus_Latn_twl.txt".format(path_toxicity_data),
    "lvs_Latn": "{}/lvs_Latn_twl.txt".format(path_toxicity_data),
    "mag_Deva": "{}/mag_Deva_twl.txt".format(path_toxicity_data),
    "mai_Deva": "{}/mai_Deva_twl.txt".format(path_toxicity_data),
    "mal_Mlym": "{}/mal_Mlym_twl.txt".format(path_toxicity_data),
    "mar_Deva": "{}/mar_Deva_twl.txt".format(path_toxicity_data),
    "min_Arab": "{}/min_Arab_twl.txt".format(path_toxicity_data),
    "min_Latn": "{}/min_Latn_twl.txt".format(path_toxicity_data),
    "mkd_Cyrl": "{}/mkd_Cyrl_twl.txt".format(path_toxicity_data),
    "mlt_Latn": "{}/mlt_Latn_twl.txt".format(path_toxicity_data),
    "mni_Beng": "{}/mni_Beng_twl.txt".format(path_toxicity_data),
    "mos_Latn": "{}/mos_Latn_twl.txt".format(path_toxicity_data),
    "mri_Latn": "{}/mri_Latn_twl.txt".format(path_toxicity_data),
    "mya_Mymr": "{}/mya_Mymr_twl.txt".format(path_toxicity_data),
    "nld_Latn": "{}/nld_Latn_twl.txt".format(path_toxicity_data),
    "nno_Latn": "{}/nno_Latn_twl.txt".format(path_toxicity_data),
    "nob_Latn": "{}/nob_Latn_twl.txt".format(path_toxicity_data),
    "npi_Deva": "{}/npi_Deva_twl.txt".format(path_toxicity_data),
    "nso_Latn": "{}/nso_Latn_twl.txt".format(path_toxicity_data),
    "nus_Latn": "{}/nus_Latn_twl.txt".format(path_toxicity_data),
    "nya_Latn": "{}/nya_Latn_twl.txt".format(path_toxicity_data),
    "oci_Latn": "{}/oci_Latn_twl.txt".format(path_toxicity_data),
    "ory_Orya": "{}/ory_Orya_twl.txt".format(path_toxicity_data),
    "pag_Latn": "{}/pag_Latn_twl.txt".format(path_toxicity_data),
    "pan_Guru": "{}/pan_Guru_twl.txt".format(path_toxicity_data),
    "pap_Latn": "{}/pap_Latn_twl.txt".format(path_toxicity_data),
    "pbt_Arab": "{}/pbt_Arab_twl.txt".format(path_toxicity_data),
    "pes_Arab": "{}/pes_Arab_twl.txt".format(path_toxicity_data),
    "plt_Latn": "{}/plt_Latn_twl.txt".format(path_toxicity_data),
    "pol_Latn": "{}/pol_Latn_twl.txt".format(path_toxicity_data),
    "por_Latn": "{}/por_Latn_twl.txt".format(path_toxicity_data),
    "prs_Arab": "{}/prs_Arab_twl.txt".format(path_toxicity_data),
    "quy_Latn": "{}/quy_Latn_twl.txt".format(path_toxicity_data),
    "ron_Latn": "{}/ron_Latn_twl.txt".format(path_toxicity_data),
    "run_Latn": "{}/run_Latn_twl.txt".format(path_toxicity_data),
    "rus_Cyrl": "{}/rus_Cyrl_twl.txt".format(path_toxicity_data),
    "sag_Latn": "{}/sag_Latn_twl.txt".format(path_toxicity_data),
    "san_Deva": "{}/san_Deva_twl.txt".format(path_toxicity_data),
    "sat_Beng": "{}/sat_Beng_twl.txt".format(path_toxicity_data),
    "sat_Olck": "{}/sat_Olck_twl.txt".format(path_toxicity_data),
    "scn_Latn": "{}/scn_Latn_twl.txt".format(path_toxicity_data),
    "shn_Mymr": "{}/shn_Mymr_twl.txt".format(path_toxicity_data),
    "sin_Sinh": "{}/sin_Sinh_twl.txt".format(path_toxicity_data),
    "slk_Latn": "{}/slk_Latn_twl.txt".format(path_toxicity_data),
    "slv_Latn": "{}/slv_Latn_twl.txt".format(path_toxicity_data),
    "smo_Latn": "{}/smo_Latn_twl.txt".format(path_toxicity_data),
    "sna_Latn": "{}/sna_Latn_twl.txt".format(path_toxicity_data),
    "snd_Arab": "{}/snd_Arab_twl.txt".format(path_toxicity_data),
    "som_Latn": "{}/som_Latn_twl.txt".format(path_toxicity_data),
    "sot_Latn": "{}/sot_Latn_twl.txt".format(path_toxicity_data),
    "spa_Latn": "{}/spa_Latn_twl.txt".format(path_toxicity_data),
    "srd_Latn": "{}/srd_Latn_twl.txt".format(path_toxicity_data),
    "srp_Cyrl": "{}/srp_Cyrl_twl.txt".format(path_toxicity_data),
    "ssw_Latn": "{}/ssw_Latn_twl.txt".format(path_toxicity_data),
    "sun_Latn": "{}/sun_Latn_twl.txt".format(path_toxicity_data),
    "swe_Latn": "{}/swe_Latn_twl.txt".format(path_toxicity_data),
    "swh_Latn": "{}/swh_Latn_twl.txt".format(path_toxicity_data),
    "szl_Latn": "{}/szl_Latn_twl.txt".format(path_toxicity_data),
    "tam_Taml": "{}/tam_Taml_twl.txt".format(path_toxicity_data),
    "taq_Latn": "{}/taq_Latn_twl.txt".format(path_toxicity_data),
    "taq_Tfng": "{}/taq_Tfng_twl.txt".format(path_toxicity_data),
    "tat_Cyrl": "{}/tat_Cyrl_twl.txt".format(path_toxicity_data),
    "tel_Telu": "{}/tel_Telu_twl.txt".format(path_toxicity_data),
    "tgk_Cyrl": "{}/tgk_Cyrl_twl.txt".format(path_toxicity_data),
    "tgl_Latn": "{}/tgl_Latn_twl.txt".format(path_toxicity_data),
    "tha_Thai": "{}/tha_Thai_twl.txt".format(path_toxicity_data),
    "tir_Ethi": "{}/tir_Ethi_twl.txt".format(path_toxicity_data),
    "tpi_Latn": "{}/tpi_Latn_twl.txt".format(path_toxicity_data),
    "tsn_Latn": "{}/tsn_Latn_twl.txt".format(path_toxicity_data),
    "tso_Latn": "{}/tso_Latn_twl.txt".format(path_toxicity_data),
    "tuk_Latn": "{}/tuk_Latn_twl.txt".format(path_toxicity_data),
    "tum_Latn": "{}/tum_Latn_twl.txt".format(path_toxicity_data),
    "tur_Latn": "{}/tur_Latn_twl.txt".format(path_toxicity_data),
    "twi_Latn": "{}/twi_Latn_twl.txt".format(path_toxicity_data),
    "tzm_Tfng": "{}/tzm_Tfng_twl.txt".format(path_toxicity_data),
    "uig_Arab": "{}/uig_Arab_twl.txt".format(path_toxicity_data),
    "ukr_Cyrl": "{}/ukr_Cyrl_twl.txt".format(path_toxicity_data),
    "umb_Latn": "{}/umb_Latn_twl.txt".format(path_toxicity_data),
    "urd_Arab": "{}/urd_Arab_twl.txt".format(path_toxicity_data),
    "uzn_Latn": "{}/uzn_Latn_twl.txt".format(path_toxicity_data),
    "vec_Latn": "{}/vec_Latn_twl.txt".format(path_toxicity_data),
    "vie_Latn": "{}/vie_Latn_twl.txt".format(path_toxicity_data),
    "war_Latn": "{}/war_Latn_twl.txt".format(path_toxicity_data),
    "wol_Latn": "{}/wol_Latn_twl.txt".format(path_toxicity_data),
    "xho_Latn": "{}/xho_Latn_twl.txt".format(path_toxicity_data),
    "ydd_Hebr": "{}/ydd_Hebr_twl.txt".format(path_toxicity_data),
    "yor_Latn": "{}/yor_Latn_twl.txt".format(path_toxicity_data),
    "yue_Hant": "{}/yue_Hant_twl.txt".format(path_toxicity_data),
    "zho_Hans": "{}/zho_Hans_twl.txt".format(path_toxicity_data),
    "zho_Hant": "{}/zho_Hant_twl.txt".format(path_toxicity_data),
    "zsm_Latn": "{}/zsm_Latn_twl.txt".format(path_toxicity_data),
    "zul_Latn": "{}/zul_Latn_twl.txt".format(path_toxicity_data),
}