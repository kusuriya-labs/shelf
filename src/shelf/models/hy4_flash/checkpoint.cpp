#include <shelf/models/hy4_flash/checkpoint.h>
#include <shelf/quantization/exl3/exl3_int8.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace shelf::hy4 {
namespace {
using Json = nlohmann::json;
Json read_json(const Path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("ファイルを開けません: " + path.string());
    return Json::parse(in);
}
void require(bool valid, const std::string& msg) {
    if (!valid) throw std::runtime_error(msg);
}
template<class T> T take(Result<T> value) {
    if (!value.ok()) throw std::runtime_error(value.status().message());
    return value.release();
}
}
Owners read_owners(const Path& file) {
    const auto data = read_json(file);
    Owners owners{};
    require(data.size()==41, "配置表の層数が違います");
    for (size_t i=0; i<41; ++i) {
        require(data[i].size()==512, "配置表の expert 数が違います");
        for (size_t e=0; e<512; ++e) {
            require(data[i][e].is_number_integer(), "配置表の rank が不正です");
            int r=data[i][e].get<int>();
            require(r>=0 && r<3, "配置表の rank が範囲外です");
            owners[i][e]=r;
        }
    }
    return owners;
}
const Weight& Checkpoint::at(const std::string& key) const {
    const auto it=weights.find(key);
    if(it==weights.end()) throw std::runtime_error("重みがありません: "+key);
    return it->second;
}
Checkpoint::Checkpoint(const Path& model,const Path& engram,const Path& draft,const Owners& layout)
    : owners(layout), has_draft(!draft.empty()) {
    const auto cfg=read_json(model/"config.json");
    require(cfg.at("model_type")=="hy_v4_flash" && cfg.at("exl3_text_only")==true &&
            cfg.at("num_hidden_layers")==42 && cfg.at("hc_mult")==4 &&
            cfg.at("n_routed_experts")==512 && cfg.at("num_experts_per_tok")==16 &&
            cfg.at("hidden_size")==3072 && cfg.at("routed_expert_hidden_size")==1536 &&
            cfg.at("kv_lora_rank")==512 && cfg.at("vocab_size")==120832,
            "Hy4-flash の text 専用 checkpoint が必要です");
    const auto& quant=cfg.at("quantization_config");
    require(quant.at("mul1_multiplier")==quantization::exl3::kExl3Mul24Multiplier &&
            quant.at("bits")==4 && quant.at("head_bits")==6, "量子化形式が不正です");
    require(std::filesystem::is_regular_file(model/"conversion_report.json"), "本体の組立が未完了です");
    for(int i=0;i<42;++i) {
        const bool sparse=i==0 || i%4==1;
        require(cfg.at("layer_types").at(i)==(sparse?"deepseek_sparse_attention":"sliding_attention"),
                "attention 層の順序が異なります");
    }
    for(const auto& [name,meta]:quant.at("linear_storage").items())
        shape[name.substr(0,name.size()-7)]=meta.at("original_shape").get<std::array<int,2>>();
    load(model,false);
    for (int g=0;g<4;++g) {
        int layer=1+10*g;
        const auto& orders=cfg.at("engram_layer_head_vocab_sizes").at(std::to_string(layer+1));
        size_t rows=0;
        for(int h=0;h<16;++h) {
            int size=orders.at(h/8).at(h%8).get<int>();
            require(size>0,"Engram の語彙数が不正です");
            rows+=size;
            moduli[g][h]=size;
        }
        auto key="model.layers."+std::to_string(layer)+".engram_layer.multi_head_embedding.embedding.weight";
        auto reader=take(io::SafetensorsReader::open((engram/(key+".fp8.safetensors")).string()));
        for(bool scale:{false,true}) {
            auto name=scale ? key+".scale" : key;
            auto spec=take(reader.tensor_spec(name));
            auto expected=scale ? io::SType::BF16 : io::SType::F8_E4M3;
            require(spec.dtype==expected && spec.shape==std::vector<size_t>{rows,scale?1u:96u},
                    "Engram 表の型または形状が不正です");
            size_t length;
            auto data=take(reader.tensor_data(name,length));
            weights.emplace(name,Weight{spec,data,length,-1,0});
        }
        files.push_back(std::move(reader));
    }
    if(has_draft) {
        auto q=read_json(draft/"conversion_report.json");
        require(q.at("method")=="dfly" && q.at("bits")==4,"DFly の変換結果が不正です");
        require(q.at("body_assembly_signature")==read_json(model/"conversion_report.json").at("assembly_signature"),
                "DFly と本体の校正元が一致しません");
        for(const auto& [name,dims]:q.at("linear_shapes").items())
            shape[name]=dims.get<std::array<int,2>>();
        load(draft,true);
    }
    for(const auto& [key,w]:weights) {
        if(!key.ends_with(".trellis")) continue;
        const auto name=key.substr(0,key.size()-8);
        const bool expert=name.find(".mlp.experts.")!=std::string::npos;
        auto logical=expert ? std::array<int,2>{1536,1536} : shape.at(name);
        int k=logical[0], n=logical[1], bits=name=="lm_head"?6:4;
        require(k>0 && n>0 && k<=120832 && n<=120832,"線形の幅が不正です: "+name);
        size_t pk=(k+127)/128*128, pn=(n+127)/128*128;
        auto matrix=[&](const std::string& suffix,io::SType dtype,std::vector<size_t> dims){
            const auto& part=at(name+suffix);
            require(part.spec.dtype==dtype && part.spec.shape==dims,"量子化 tensor が不正です: "+name+suffix);
        };
        matrix(".suh",io::SType::F16,{pk});
        matrix(".svh",io::SType::F16,{pn});
        matrix(".trellis",io::SType::I16,{pk/16,pn/16,size_t(16*bits)});
        matrix(".mul1",io::SType::I32,{});
        int32_t marker;
        std::memcpy(&marker,at(name+".mul1").host,sizeof(marker));
        require(marker==quantization::exl3::kExl3Mul24Multiplier,"mul24 乗数が不正です");
    }
    for(const auto& [name,dims]:shape) at(name+".trellis");
    for(int i=1;i<42;++i) for(int e=0;e<512;++e) {
        auto stem="model.layers."+std::to_string(i)+".mlp.experts."+std::to_string(e)+".";
        for(auto name:{"gate_proj","up_proj","down_proj"}) at(stem+name+".trellis");
    }
}
void Checkpoint::load(const Path& directory,bool draft) {
    auto index=read_json(directory/"model.safetensors.index.json").at("weight_map");
    std::map<std::string,std::vector<std::string>> shards;
    for(const auto& [key,file]:index.items()) {
        require(key.starts_with("model.dfly_layers.")==draft,"本体・DFly の index が混在しています");
        if(key.ends_with(".multi_head_embedding.embedding.weight")) continue;
        shards[file.get<std::string>()].push_back(key);
    }
    for(const auto& [file,keys]:shards) {
        require(Path(file).filename()==file,"shard ファイル名が不正です");
        auto reader=take(io::SafetensorsReader::open((directory/file).string()));
        for(const auto& key:keys) {
            int rank=0;
            if(key.find(".mlp.experts.")!=std::string::npos) {
                int layer=0,e=0;
                require(sscanf(key.c_str(),"model.layers.%d.mlp.experts.%d.",&layer,&e)==2 &&
                        layer>=1 && layer<=41 && e>=0 && e<512,"expert 名が不正です");
                rank=owners[layer-1][e];
            }
            if(key.ends_with(".mul1") || key.ends_with(".lookup_table")) rank=-1;
            size_t length;
            auto ptr=take(reader.tensor_data(key,length));
            auto spec=take(reader.tensor_spec(key));
            size_t offset=0;
            if(rank>=0) {
                offset=(bytes[rank]+15)&~size_t(15);
                bytes[rank]=offset+length;
            }
            require(weights.emplace(key,Weight{std::move(spec),ptr,length,rank,offset}).second,"tensor 名の重複");
        }
        files.push_back(std::move(reader));
    }
}
void Checkpoint::relocate(const Owners& layout) {
    bytes.fill(0);
    for(auto& [key,w]:weights) {
        if(w.rank<0) continue;
        if(key.find(".mlp.experts.")!=std::string::npos) {
            int layer=0,e=0;
            if(sscanf(key.c_str(),"model.layers.%d.mlp.experts.%d.",&layer,&e)!=2)
                throw std::runtime_error("expert 名の解析に失敗しました");
            w.rank=layout.at(layer-1).at(e);
        }
        w.offset=(bytes[w.rank]+15)&~size_t(15);
        bytes[w.rank]=w.offset+w.bytes;
    }
    owners=layout;
}

}
