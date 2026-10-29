# shelf — 「1 ディレクトリ ≒ 1 ターゲット」の整合確認（D7）
#
# 目的: モジュールディレクトリと CMake ターゲットの対応が崩れたら **configure で止める**。
#       （テストではなく configure 時に見るのは、ターゲット定義そのものを検査対象に
#        するため。ctest からは再現できない。）
#
# 検査内容:
#  1. マニフェストの各ディレクトリについて
#       - ターゲットがある → そのソースは**すべてそのディレクトリ配下**にあり、1 つ以上ある
#         （INTERFACE ターゲットなら、そのディレクトリにコンパイル対象のソースが無い）
#       - ターゲットが無い → そのディレクトリにコンパイル対象のソースが無い
#  2. ルートスコープの `shelf_*` ターゲットは、必ずマニフェストに載っている
#  3. `src/` 配下でソースを持つディレクトリは、必ずどれか 1 つのマニフェスト配下にある
#     （= 宣言されていない場所にコードを置けない）
#
# 追加するとき: 下の SHELF_MODULE_TARGETS に 1 行足し、同時に
#   draft_docs/dependency_rules.md §1 の対応表と tests/architecture のルールを更新する。

set(SHELF_MODULE_TARGETS
    # ディレクトリ（PROJECT_SOURCE_DIR からの相対）: ターゲット名
    "src/foundation:shelf_foundation"
    "src/ir:shelf_ir"                          # INTERFACE（ヘッダのみ）
    "src/artifact:shelf_artifact"
    "src/quant:shelf_quant"
    "src/weights:shelf_weights"                # INTERFACE（ヘッダのみ）
    "src/model/qwen35:shelf_model_qwen35"
    "src/runtime/state:shelf_runtime_state"
    "src/runtime/boot:shelf_runtime_boot"      # 未作成（D1）
    "src/runtime/pipeline:shelf_runtime_pipeline"
    "src/runtime/load:shelf_runtime_load"      # D3
    "src/compiler:shelf_compiler"              # 未作成（D4）
    "src/placement:shelf_placement"            # 未作成（D2）
    "src/backend/common:shelf_backend_common"
    "src/backend/cpu:shelf_backend_cpu"
    "src/backend/hip:shelf_backend_hip:${SHELF_ENABLE_HIP}"   # HIP 無効ビルドでは作らない
    "src/api:shelf_api"                        # 未作成（D8）
)

set(_shelf_src_dir "${PROJECT_SOURCE_DIR}/src")

# コンパイル対象のソース拡張子（ヘッダは「ソース」に数えない）。
function(_shelf_dir_sources out_var dir)
    set(_found "")
    if(IS_DIRECTORY "${dir}")
        file(GLOB_RECURSE _files
             "${dir}/*.cpp" "${dir}/*.cc" "${dir}/*.cxx" "${dir}/*.hip" "${dir}/*.cu")
        foreach(_f IN LISTS _files)
            if(NOT _f MATCHES "/include/")
                list(APPEND _found "${_f}")
            endif()
        endforeach()
    endif()
    set(${out_var} "${_found}" PARENT_SCOPE)
endfunction()

# (2) ルートスコープの shelf_* ターゲット一覧
get_property(_root_targets DIRECTORY "${PROJECT_SOURCE_DIR}" PROPERTY BUILDSYSTEM_TARGETS)

set(_declared_targets "")
set(_declared_dirs "")
foreach(_entry IN LISTS SHELF_MODULE_TARGETS)
    string(REPLACE ":" ";" _pair "${_entry}")
    list(GET _pair 0 _dir_rel)
    list(GET _pair 1 _target)
    # 3 つ目は「ターゲットを作る条件」（空なら常に必須）。
    list(LENGTH _pair _pair_len)
    set(_cond "")
    if(_pair_len GREATER 2)
        list(GET _pair 2 _cond)
    endif()
    set(_required TRUE)
    if(NOT _cond STREQUAL "" AND NOT ${_cond})
        set(_required FALSE)
    endif()
    set(_dir "${PROJECT_SOURCE_DIR}/${_dir_rel}")
    list(APPEND _declared_targets "${_target}")
    list(APPEND _declared_dirs "${_dir_rel}")

    _shelf_dir_sources(_dir_srcs "${_dir}")

    if(TARGET "${_target}")
        get_target_property(_kind "${_target}" TYPE)
        if(_kind STREQUAL "INTERFACE_LIBRARY")
            if(_dir_srcs)
                message(FATAL_ERROR
                    "shelf: ${_dir_rel} は INTERFACE ターゲット ${_target} だが、"
                    "コンパイル対象のソースがある（${_dir_srcs}）")
            endif()
        else()
            get_target_property(_srcs "${_target}" SOURCES)
            if(NOT _srcs)
                message(FATAL_ERROR
                    "shelf: ターゲット ${_target} にソースが無い（${_dir_rel}）")
            endif()
            foreach(_s IN LISTS _srcs)
                if(NOT IS_ABSOLUTE "${_s}")
                    set(_abs "${PROJECT_SOURCE_DIR}/${_s}")
                else()
                    set(_abs "${_s}")
                endif()
                file(RELATIVE_PATH _rel "${_dir}" "${_abs}")
                if(_rel MATCHES "^\\.\\.")
                    message(FATAL_ERROR
                        "shelf: ターゲット ${_target} のソース ${_abs} が"
                        "${_dir_rel} の外にある（1 ディレクトリ ≒ 1 ターゲット違反）")
                endif()
            endforeach()
        endif()
    elseif(_required)
        if(_dir_srcs)
            message(FATAL_ERROR
                "shelf: ${_dir_rel} にソースがあるが、ターゲット ${_target} が定義されていない"
                "（cmake/Targets.cmake を確認）")
        endif()
    endif()
endforeach()

# (2) 逆方向: `src/` のソースを持つのに宣言されていないターゲットが無いか。
#     外部依存のラッパー（shelf_nlohmann 等）は src/ の外なので対象外。
foreach(_t IN LISTS _root_targets)
    if(_t IN_LIST _declared_targets)
        continue()
    endif()
    get_target_property(_t_srcs "${_t}" SOURCES)
    if(NOT _t_srcs)
        continue()   # INTERFACE / ヘッダのみのターゲット
    endif()
    foreach(_s IN LISTS _t_srcs)
        if(NOT IS_ABSOLUTE "${_s}")
            set(_abs "${PROJECT_SOURCE_DIR}/${_s}")
        else()
            set(_abs "${_s}")
        endif()
        string(FIND "${_abs}" "${PROJECT_SOURCE_DIR}/src/" _pos)
        if(_pos EQUAL 0)
            message(FATAL_ERROR
                "shelf: ターゲット ${_t}（src/ のソースを持つ）が SHELF_MODULE_TARGETS に無い"
                "（cmake/CheckModuleTargets.cmake に 1 行足す）")
        endif()
    endforeach()
endforeach()

# (3) 宣言されていない場所にソースが無いか
file(GLOB_RECURSE _all_srcs
     "${_shelf_src_dir}/*.cpp" "${_shelf_src_dir}/*.cc" "${_shelf_src_dir}/*.cxx"
     "${_shelf_src_dir}/*.hip" "${_shelf_src_dir}/*.cu")
foreach(_f IN LISTS _all_srcs)
    if(_f MATCHES "/include/")
        continue()
    endif()
    set(_covered FALSE)
    foreach(_d IN LISTS _declared_dirs)
        set(_prefix "${PROJECT_SOURCE_DIR}/${_d}/")
        string(FIND "${_f}" "${_prefix}" _pos)
        if(_pos EQUAL 0)
            set(_covered TRUE)
            break()
        endif()
    endforeach()
    if(NOT _covered)
        message(FATAL_ERROR
            "shelf: ${_f} が宣言されたモジュールディレクトリの外にある"
            "（SHELF_MODULE_TARGETS に追加するか、既存モジュールへ移す）")
    endif()
endforeach()

list(LENGTH SHELF_MODULE_TARGETS _n_dirs)
message(STATUS "shelf: モジュールとターゲットの整合 OK（${_n_dirs} ディレクトリ）")
