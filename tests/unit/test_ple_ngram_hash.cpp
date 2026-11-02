#include <shelf/models/qwen4exp/ple/ngram_hash.h>

#include <cstdint>
#include <cstdio>
#include <vector>

namespace ple = ps::qwen4exp::ple;

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* msg) {
    if (cond) {
        ++passed;
        printf("PASS: %s\n", msg);
    } else {
        ++failed;
        printf("FAIL: %s\n", msg);
    }
}

static constexpr std::uint32_t kEos = 248044;
static constexpr std::uint64_t kMaxInt63 = 0x7FFFFFFFFFFFFFFFull;

static const std::int64_t kGoldenSizesSlot0[16] = {
    20000003ll, 20000023ll, 20000033ll, 20000047ll, 20000059ll, 20000063ll, 20000069ll, 20000077ll,
    20000081ll, 20000093ll, 20000107ll, 20000147ll, 20000153ll, 20000159ll, 20000161ll, 20000171ll,
};
static const std::int64_t kGoldenOffsetsSlot0[16] = {
    0ll, 20000003ll, 40000026ll, 60000059ll, 80000106ll, 100000165ll, 120000228ll, 140000297ll,
    160000374ll, 180000455ll, 200000548ll, 220000655ll, 240000802ll, 260000955ll, 280001114ll,
    300001275ll,
};
static const std::int64_t kGoldenMultipliersSlot0[3] = {
    23703573157769ll, 20109073645365ll, 8052911324071ll,
};
static const std::int64_t kGoldenSizesSlot1[16] = {
    20000213ll, 20000221ll, 20000243ll, 20000269ll, 20000287ll, 20000297ll, 20000303ll, 20000311ll,
    20000327ll, 20000329ll, 20000339ll, 20000353ll, 20000359ll, 20000377ll, 20000389ll, 20000429ll,
};
static const std::int64_t kGoldenMultipliersSlot1[3] = {
    3352040966061ll, 19752442147475ll, 10742103100073ll,
};
static const std::uint64_t kSplitmixIn[10] = {
    0ull, 1ull, 11400714819323198485ull, 18446744073709551615ull, 18446744073709551599ull, 1234ull,
    11241ull, 11400714819323199719ull, 15755400384260045073ull, 9223372036854775808ull,
};
static const std::uint64_t kSplitmixOut[10] = {
    16294208416658607535ull, 10451216379200822465ull, 7960286522194355700ull,
    16490336266968443936ull, 15719791520248931999ull, 13478418381427711195ull,
    1257261698672413609ull, 10936887474700444964ull, 5648149391703318579ull, 5196802822362493915ull,
};
static const std::uint64_t kGoldenIdsA[48] = {
    3910472ull, 28010779ull, 50061374ull, 60932650ull, 87394228ull, 116214909ull, 129445926ull,
    147087442ull, 175402443ull, 195013446ull, 207893337ull, 233266953ull, 253073443ull,
    272880023ull, 286148956ull, 312493449ull, 15567690ull, 35687509ull, 55747620ull, 67831959ull,
    83904455ull, 115928743ull, 123965091ull, 148013673ull, 174920563ull, 193749848ull, 219050836ull,
    228482654ull, 247897655ull, 267312686ull, 293784530ull, 306142834ull, 9319853ull, 32980418ull,
    54810932ull, 61373850ull, 95570932ull, 100303321ull, 137402047ull, 146866965ull, 160487518ull,
    192851935ull, 200610520ull, 228493582ull, 254676383ull, 260859079ull, 296253588ull,
    313224935ull,
};
static const std::uint64_t kGoldenIdsB[48] = {
    3910472ull, 28010779ull, 50061374ull, 60932650ull, 87394228ull, 116214909ull, 129445926ull,
    147087442ull, 175402443ull, 195013446ull, 207893337ull, 233266953ull, 253073443ull,
    272880023ull, 286148956ull, 312493449ull, 18392917ull, 34857659ull, 47499630ull, 74137148ull,
    90126688ull, 103063933ull, 133351754ull, 142048315ull, 170176673ull, 193347574ull, 202397286ull,
    220003112ull, 252701089ull, 266457241ull, 291277968ull, 317144678ull, 2927653ull, 34980843ull,
    54748278ull, 66612378ull, 97814964ull, 109013870ull, 126560393ull, 151352333ull, 167888935ull,
    182235580ull, 215170017ull, 237467519ull, 247510681ull, 278779700ull, 296141806ull,
    304994522ull,
};
static const std::uint64_t kGoldenIdsC[16] = {
    18043673ull, 37626835ull, 51159316ull, 78294604ull, 94015356ull, 106720349ull, 136526052ull,
    144330141ull, 176817901ull, 186368539ull, 203707490ull, 230017629ull, 247662678ull,
    266533413ull, 293096193ull, 307951937ull,
};
static const std::uint64_t kGoldenIdsD[32] = {
    6925160ull, 25283190ull, 54477378ull, 67366163ull, 98429496ull, 102120486ull, 127660064ull,
    155051809ull, 162064539ull, 183784834ull, 214294501ull, 229502072ull, 254175458ull,
    279212006ull, 287638224ull, 310374413ull, 9058818ull, 22549699ull, 59295284ull, 70739163ull,
    94834058ull, 109532400ull, 131579937ull, 140976624ull, 177983952ull, 191505481ull, 210621367ull,
    222424812ull, 259201214ull, 275978939ull, 281571840ull, 309538795ull,
};
static const std::uint64_t kGoldenIdsE[16] = {
    3035822ull, 30922970ull, 45995374ull, 64361017ull, 92705579ull, 115727959ull, 130487234ull,
    157254389ull, 171350132ull, 183732882ull, 212505503ull, 229174638ull, 257427880ull,
    265877325ull, 282070880ull, 303365432ull,
};
static const std::uint32_t kGoldenShift1[5] = {
    248044, 11, 22, 5, 248044,
};
static const std::uint32_t kGoldenShift2[5] = {
    248044, 248044, 11, 22, 248044,
};
static const std::uint64_t kGoldenIdsWrap[48] = {
    2544533ull, 22474812ull, 52816253ull, 70915639ull, 80077801ull, 105345514ull, 135322308ull,
    152498408ull, 160762411ull, 199455261ull, 217558349ull, 235206744ull, 247627076ull,
    261292546ull, 299458001ull, 312359067ull, 7943010ull, 27896537ull, 54790842ull, 60190417ull,
    86298576ull, 103143718ull, 129794935ull, 147912358ull, 175406026ull, 186560544ull, 216732348ull,
    239479624ull, 246846744ull, 276289249ull, 293231301ull, 301399627ull, 2519646ull, 22449925ull,
    52791366ull, 70890752ull, 80052914ull, 105320627ull, 135297421ull, 152473521ull, 170050241ull,
    193666601ull, 215907028ull, 223753038ull, 246067093ull, 271286474ull, 287005282ull,
    310441039ull,
};
static const std::int64_t kWrapMultipliers[3] = {
    4611686018427400249ll, 9223372036854775805ll, 2305843009213693959ll,
};
static const std::uint32_t kRandHist0[5] = {
    190332, 19460, 80805, 170144, 4922,
};
static const std::uint64_t kRandIds0[48] = {
    17991533ull, 24345058ull, 59250085ull, 78052696ull, 87395119ull, 110878007ull, 126447924ull,
    154519711ull, 161251033ull, 180690691ull, 204075164ull, 235070450ull, 246254243ull,
    277577617ull, 281383006ull, 300642321ull, 13933441ull, 39951487ull, 54768630ull, 69537703ull,
    86934494ull, 106452529ull, 126091176ull, 146284381ull, 176818780ull, 185989946ull, 212340679ull,
    229952534ull, 259351824ull, 269209397ull, 285930568ull, 310300000ull, 15117452ull, 39278466ull,
    53893333ull, 65192599ull, 91799160ull, 107875364ull, 132496544ull, 146270875ull, 167493365ull,
    189281828ull, 207317065ull, 222937387ull, 249263699ull, 276107349ull, 291836916ull,
    311346488ull,
};
static const std::uint32_t kRandHist1[6] = {
    137918, 227135, 248044, 16134, 194585, 171538,
};
static const std::uint64_t kRandIds1[64] = {
    8267470ull, 32749265ull, 40987811ull, 63239150ull, 97120739ull, 109694102ull, 139753710ull,
    148738859ull, 179944278ull, 185262677ull, 214532458ull, 221841518ull, 247789414ull,
    275002896ull, 297688738ull, 313226443ull, 5698148ull, 23421209ull, 45766451ull, 68951514ull,
    86733166ull, 106736944ull, 137439355ull, 159676418ull, 162743420ull, 188723740ull, 211720648ull,
    224576677ull, 250069757ull, 276753591ull, 292579534ull, 313693346ull, 7429846ull, 33985097ull,
    40965251ull, 70884335ull, 94665609ull, 116715949ull, 130531904ull, 157002147ull, 171934834ull,
    189060137ull, 218617384ull, 237263368ull, 248795015ull, 261561804ull, 292758757ull,
    310800975ull, 12313780ull, 35659115ull, 58095364ull, 66361263ull, 82811868ull, 108458358ull,
    127080748ull, 158862363ull, 175060514ull, 189540941ull, 217479219ull, 234928555ull,
    254337971ull, 273953973ull, 287205268ull, 313805731ull,
};
static const std::uint32_t kRandHist2[8] = {
    230024, 134287, 223375, 243768, 170727, 131902, 160766, 239167,
};
static const std::uint64_t kRandIds2[96] = {
    2872325ull, 33320005ull, 54381235ull, 70404784ull, 87352965ull, 100914349ull, 122423899ull,
    153282587ull, 171877862ull, 180958360ull, 205937886ull, 237397656ull, 257969479ull,
    260067957ull, 294440340ull, 308845657ull, 18263985ull, 22435022ull, 40486932ull, 72441960ull,
    86036969ull, 118508212ull, 138408262ull, 147169033ull, 169937896ull, 199550070ull, 207151128ull,
    235340497ull, 243411771ull, 272746558ull, 289472288ull, 315206055ull, 3935571ull, 30629307ull,
    50687345ull, 78285087ull, 91776993ull, 117706066ull, 127941811ull, 144094959ull, 165754081ull,
    194788568ull, 218608124ull, 220189307ull, 258440747ull, 278000046ull, 291477200ull,
    301042487ull, 13829799ull, 37238986ull, 59201721ull, 70238617ull, 97110163ull, 106122413ull,
    139692481ull, 157882189ull, 167613734ull, 189007635ull, 219396951ull, 235806142ull,
    240857096ull, 266322875ull, 294903779ull, 318498463ull, 4122288ull, 29763120ull, 43457786ull,
    67609485ull, 97791618ull, 108038837ull, 133584549ull, 154638511ull, 177958235ull, 196334257ull,
    204650954ull, 221095911ull, 250723149ull, 260392054ull, 290291195ull, 319855644ull, 5212756ull,
    31481551ull, 59936230ull, 61731388ull, 94517497ull, 106581188ull, 135740808ull, 143273077ull,
    160125475ull, 192066308ull, 219798031ull, 235651034ull, 256438464ull, 278637040ull,
    293016873ull, 307267455ull,
};
static const std::uint32_t kRandHist3[8] = {
    104451, 131789, 41888, 198580, 45675, 248044, 113968, 235185,
};
static const std::uint64_t kRandIds3[96] = {
    13600163ull, 39843332ull, 45180735ull, 67134843ull, 93971417ull, 110056340ull, 124626859ull,
    158214845ull, 177896368ull, 181099895ull, 210444301ull, 227776193ull, 244099931ull,
    260873375ull, 286564618ull, 315769548ull, 4206769ull, 30701409ull, 47995719ull, 76740379ull,
    85587468ull, 116066599ull, 132594601ull, 156142809ull, 166123602ull, 189424136ull, 207756181ull,
    238155084ull, 259113049ull, 260957301ull, 295102638ull, 307305445ull, 18064720ull, 26472887ull,
    43726312ull, 71296365ull, 98099151ull, 107683937ull, 122670980ull, 143792120ull, 178260112ull,
    187306825ull, 211845616ull, 237027372ull, 259825765ull, 263412093ull, 298116266ull,
    312949333ull, 7145713ull, 30962319ull, 47904942ull, 69263031ull, 95662798ull, 118870081ull,
    124687771ull, 154324232ull, 170109230ull, 191886346ull, 208883148ull, 238090391ull,
    259204809ull, 261292958ull, 295539026ull, 308391228ull, 4545565ull, 36368345ull, 57482625ull,
    72869804ull, 88612645ull, 101636889ull, 122213829ull, 151592157ull, 179935433ull, 184365214ull,
    214330368ull, 233904568ull, 243005428ull, 273714633ull, 297641852ull, 319957462ull, 11875797ull,
    31331753ull, 45522401ull, 62387477ull, 84341574ull, 105945015ull, 129242681ull, 155305610ull,
    175228921ull, 195940234ull, 214901158ull, 226116433ull, 256436762ull, 267705946ull,
    285006670ull, 313091487ull,
};

static bool is_prime_i64(std::int64_t n) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2;
    for (std::int64_t d = 3; d * d <= n; d += 2) {
        if (n % d == 0) return false;
    }
    return true;
}

static void test_splitmix64_vectors() {
    bool ok = true;
    for (std::size_t i = 0; i < sizeof(kSplitmixIn) / sizeof(kSplitmixIn[0]); ++i) {
        const std::uint64_t got = ple::splitmix64(kSplitmixIn[i]);
        if (got != kSplitmixOut[i]) {
            ok = false;
            printf("  splitmix64 mismatch in=%llu got=%llu want=%llu\n",
                   static_cast<unsigned long long>(kSplitmixIn[i]),
                   static_cast<unsigned long long>(got),
                   static_cast<unsigned long long>(kSplitmixOut[i]));
        }
    }
    check(ok, "splitmix64: oracle vectors incl. 64-bit wrap");
}

static void test_tables_slot0() {
    ple::NgramHashParams params;
    params.seed = 1234;
    params.unigram_vocab_size = 248320;
    params.ple_layer_slot = 0;

    ple::NgramIdTables tables;
    const ps::Status st = ple::build_ngram_id_tables(params, tables);
    check(st.ok(), "tables: build slot0");
    if (!st.ok()) {
        printf("  reason: %s\n", st.message().c_str());
        return;
    }

    bool sizes_ok = true;
    bool prime_ok = true;
    for (std::uint32_t h = 0; h < ple::kNgramHeads; ++h) {
        if (tables.vocab_sizes[h] != kGoldenSizesSlot0[h]) sizes_ok = false;
        if (!is_prime_i64(tables.vocab_sizes[h])) prime_ok = false;
    }
    check(sizes_ok, "tables: slot0 vocab sizes equal Python oracle");
    check(prime_ok, "tables: slot0 vocab sizes are prime");

    bool offsets_ok = true;
    std::int64_t acc = 0;
    for (std::uint32_t h = 0; h < ple::kNgramHeads; ++h) {
        if (tables.offsets[h] != kGoldenOffsetsSlot0[h]) offsets_ok = false;
        if (tables.offsets[h] != acc) offsets_ok = false;
        acc += tables.vocab_sizes[h];
    }
    check(offsets_ok, "tables: slot0 offsets are cumulative sums");

    bool mult_ok = true;
    bool odd_ok = true;
    for (std::uint32_t k = 0; k < ple::kNgramSize; ++k) {
        if (tables.multipliers[k] != kGoldenMultipliersSlot0[k]) mult_ok = false;
        if (tables.multipliers[k] <= 0 || (tables.multipliers[k] % 2) == 0) odd_ok = false;
    }
    check(mult_ok, "tables: slot0 multipliers equal Python oracle");
    check(odd_ok, "tables: multipliers are positive odd");

    const std::uint64_t max_token = 248319ull;
    std::uint64_t max_mult = 0;
    for (std::uint32_t k = 0; k < ple::kNgramSize; ++k) {
        const std::uint64_t m = static_cast<std::uint64_t>(tables.multipliers[k]);
        if (m > max_mult) max_mult = m;
    }
    check(max_token * max_mult < kMaxInt63, "tables: token*multiplier fits int63");

    const std::uint64_t total_rows =
        static_cast<std::uint64_t>(tables.offsets[ple::kNgramHeads - 1]) +
        static_cast<std::uint64_t>(tables.vocab_sizes[ple::kNgramHeads - 1]);
    printf("  slot0 total rows: %llu\n", static_cast<unsigned long long>(total_rows));
    check(total_rows <= 128ull * 2500012ull, "tables: rows fit 128 shards of 2500012");
}

static void test_tables_slot1() {
    ple::NgramHashParams params;
    params.seed = 1234;
    params.unigram_vocab_size = 248320;
    params.ple_layer_slot = 1;

    ple::NgramIdTables tables;
    const ps::Status st = ple::build_ngram_id_tables(params, tables);
    check(st.ok(), "tables: build slot1");
    if (!st.ok()) return;

    bool sizes_ok = true;
    for (std::uint32_t h = 0; h < ple::kNgramHeads; ++h) {
        if (tables.vocab_sizes[h] != kGoldenSizesSlot1[h]) sizes_ok = false;
        if (!is_prime_i64(tables.vocab_sizes[h])) sizes_ok = false;
    }
    check(sizes_ok, "tables: slot1 vocab sizes equal Python oracle");

    bool mult_ok = true;
    for (std::uint32_t k = 0; k < ple::kNgramSize; ++k) {
        if (tables.multipliers[k] != kGoldenMultipliersSlot1[k]) mult_ok = false;
    }
    check(mult_ok, "tables: slot1 multipliers equal Python oracle");

    ple::NgramHashParams params20m = params;
    params20m.unigram_vocab_size = 20000000;
    std::int64_t mult20m[ple::kNgramSize];
    check(ple::build_layer_multipliers(params20m, mult20m).ok(),
          "tables: ngram_vocab_size_base variant builds");
    bool differs = false;
    for (std::uint32_t k = 0; k < ple::kNgramSize; ++k) {
        if (mult20m[k] != tables.multipliers[k]) differs = true;
    }
    check(differs, "tables: half_bound depends on unigram_vocab_size");
}

static void test_shift_golden() {
    const std::uint32_t history[5] = {11, 22, 5, kEos, 7};
    std::uint32_t out[5];
    check(ple::shift_right_ignore_eos(history, 5, 1, kEos, out).ok(), "shift: k=1 runs");
    bool ok1 = true;
    for (std::uint32_t i = 0; i < 5; ++i) {
        if (out[i] != kGoldenShift1[i]) ok1 = false;
    }
    check(ok1, "shift: k=1 matches oracle across EOS boundary");

    check(ple::shift_right_ignore_eos(history, 5, 2, kEos, out).ok(), "shift: k=2 runs");
    bool ok2 = true;
    for (std::uint32_t i = 0; i < 5; ++i) {
        if (out[i] != kGoldenShift2[i]) ok2 = false;
    }
    check(ok2, "shift: k=2 matches oracle across EOS boundary");
}

static void reference_ids(const ple::NgramIdTables& tables,
                          const std::uint32_t* context, std::uint32_t context_len,
                          const std::uint32_t* input, std::uint32_t count,
                          std::uint32_t eos, std::uint64_t* out) {
    const std::uint32_t total = context_len + count;
    std::vector<std::uint32_t> history(total);
    for (std::uint32_t i = 0; i < context_len; ++i) history[i] = context[i];
    for (std::uint32_t i = 0; i < count; ++i) history[context_len + i] = input[i];

    std::vector<std::uint32_t> shifted[ple::kNgramSize];
    for (std::uint32_t k = 0; k < ple::kNgramSize; ++k) {
        shifted[k].resize(total);
        ple::shift_right_ignore_eos(history.data(), total, k, eos, shifted[k].data());
    }
    for (std::uint32_t p = 0; p < count; ++p) {
        const std::uint32_t pos = context_len + p;
        for (std::uint32_t degree = 2; degree <= ple::kNgramSize; ++degree) {
            std::uint64_t h =
                shifted[0][pos] * static_cast<std::uint64_t>(tables.multipliers[0]);
            for (std::uint32_t k = 1; k < degree; ++k) {
                h ^= shifted[k][pos] * static_cast<std::uint64_t>(tables.multipliers[k]);
            }
            for (std::uint32_t j = 0; j < ple::kHeadsPerNgram; ++j) {
                const std::uint32_t head = (degree - 2) * ple::kHeadsPerNgram + j;
                out[p * ple::kNgramHeads + head] =
                    (h % static_cast<std::uint64_t>(tables.vocab_sizes[head])) +
                    static_cast<std::uint64_t>(tables.offsets[head]);
            }
        }
    }
}

static bool ids_equal(const std::uint64_t* a, const std::uint64_t* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            printf("  id mismatch at %zu: got=%llu want=%llu\n", i,
                   static_cast<unsigned long long>(a[i]),
                   static_cast<unsigned long long>(b[i]));
            return false;
        }
    }
    return true;
}

static ple::NgramIdTables slot0_tables() {
    ple::NgramHashParams params;
    params.seed = 1234;
    params.unigram_vocab_size = 248320;
    params.ple_layer_slot = 0;
    ple::NgramIdTables tables;
    ple::build_ngram_id_tables(params, tables);
    return tables;
}

static void test_ids_golden_cases() {
    const ple::NgramIdTables tables = slot0_tables();

    {
        const std::uint32_t context[2] = {11, 22};
        const std::uint32_t input[3] = {5, 6, 7};
        std::uint64_t ids[3 * ple::kNgramHeads];
        check(ple::compute_ngram_ids(tables, context, 2, input, 3, kEos, ids).ok(),
              "ids: case A runs");
        check(ids_equal(ids, kGoldenIdsA, 3 * ple::kNgramHeads),
              "ids: case A matches Python oracle");
        std::uint64_t ref[3 * ple::kNgramHeads];
        reference_ids(tables, context, 2, input, 3, kEos, ref);
        check(ids_equal(ids, ref, 3 * ple::kNgramHeads),
              "ids: case A matches in-test reference loop");
    }
    {
        const std::uint32_t context[2] = {11, 22};
        const std::uint32_t input[3] = {5, kEos, 7};
        std::uint64_t ids[3 * ple::kNgramHeads];
        check(ple::compute_ngram_ids(tables, context, 2, input, 3, kEos, ids).ok(),
              "ids: case B runs");
        check(ids_equal(ids, kGoldenIdsB, 3 * ple::kNgramHeads),
              "ids: case B matches Python oracle with mid-sequence EOS");
        bool first_row_same = true;
        for (std::uint32_t h = 0; h < ple::kNgramHeads; ++h) {
            if (ids[h] != kGoldenIdsA[h]) first_row_same = false;
        }
        check(first_row_same, "ids: EOS at position 1 leaves position 0 untouched");
    }
    {
        const std::uint32_t input[1] = {9};
        std::uint64_t ids[ple::kNgramHeads];
        check(ple::compute_ngram_ids(tables, nullptr, 0, input, 1, kEos, ids).ok(),
              "ids: case C first token runs");
        check(ids_equal(ids, kGoldenIdsC, ple::kNgramHeads),
              "ids: case C matches Python oracle");
        std::uint64_t ref[ple::kNgramHeads];
        reference_ids(tables, nullptr, 0, input, 1, kEos, ref);
        check(ids_equal(ids, ref, ple::kNgramHeads), "ids: case C reference loop");
    }
    {
        const std::uint32_t context[1] = {1000};
        const std::uint32_t input[2] = {9, 10};
        std::uint64_t ids[2 * ple::kNgramHeads];
        check(ple::compute_ngram_ids(tables, context, 1, input, 2, kEos, ids).ok(),
              "ids: case D runs");
        check(ids_equal(ids, kGoldenIdsD, 2 * ple::kNgramHeads),
              "ids: case D matches Python oracle");
    }
    {
        const std::uint32_t context[2] = {248319, 248319};
        const std::uint32_t input[1] = {248319};
        std::uint64_t ids[ple::kNgramHeads];
        check(ple::compute_ngram_ids(tables, context, 2, input, 1, kEos, ids).ok(),
              "ids: boundary token runs");
        check(ids_equal(ids, kGoldenIdsE, ple::kNgramHeads),
              "ids: boundary token matches oracle without overflow");
    }
    {
        ple::NgramIdTables wrap = tables;
        for (std::uint32_t k = 0; k < ple::kNgramSize; ++k) {
            wrap.multipliers[k] = kWrapMultipliers[k];
        }
        const std::uint32_t context[1] = {7};
        const std::uint32_t input[3] = {99, 100, 101};
        std::uint64_t ids[3 * ple::kNgramHeads];
        check(ple::compute_ngram_ids(wrap, context, 1, input, 3, 7, ids).ok(),
              "ids: 64-bit wrap multipliers run");
        check(ids_equal(ids, kGoldenIdsWrap, 3 * ple::kNgramHeads),
              "ids: 64-bit wrap matches Python oracle");
    }
}

static void test_random_cases() {
    const ple::NgramIdTables tables = slot0_tables();
    struct Case {
        const std::uint32_t* history;
        std::uint32_t len;
        const std::uint64_t* ids;
    };
    const Case cases[] = {
        {kRandHist0, 5, kRandIds0},
        {kRandHist1, 6, kRandIds1},
        {kRandHist2, 8, kRandIds2},
        {kRandHist3, 8, kRandIds3},
    };
    bool ok = true;
    for (const Case& c : cases) {
        const std::uint32_t count = c.len - 2u;
        std::vector<std::uint64_t> ids(count * ple::kNgramHeads);
        const ps::Status st = ple::compute_ngram_ids(tables, c.history, 2, c.history + 2,
                                                     count, kEos, ids.data());
        if (!st.ok()) {
            ok = false;
            continue;
        }
        if (!ids_equal(ids.data(), c.ids, count * ple::kNgramHeads)) ok = false;
        std::vector<std::uint64_t> ref(count * ple::kNgramHeads);
        reference_ids(tables, c.history, 2, c.history + 2, count, kEos, ref.data());
        if (!ids_equal(ids.data(), ref.data(), ref.size())) ok = false;
    }
    check(ok, "ids: random cases match Python oracle and reference loop");
}

static void test_prefill_split_and_decode() {
    const ple::NgramIdTables tables = slot0_tables();
    const std::uint32_t full_input[4] = {5, 6, 7, 8};
    std::uint64_t full_ids[4 * ple::kNgramHeads];
    const std::uint32_t context0[2] = {11, 22};
    check(ple::compute_ngram_ids(tables, context0, 2, full_input, 4, kEos, full_ids).ok(),
          "split: full prefill runs");

    {
        std::uint32_t context[2] = {11, 22};
        std::uint64_t part1[2 * ple::kNgramHeads];
        check(ple::compute_ngram_ids(tables, context, 2, full_input, 2, kEos, part1).ok(),
              "split: first chunk runs");
        std::uint32_t hist[4] = {11, 22, 5, 6};
        ple::advance_ngram_context(hist, 4, context);
        std::uint64_t part2[2 * ple::kNgramHeads];
        check(ple::compute_ngram_ids(tables, context, 2, full_input + 2, 2, kEos, part2).ok(),
              "split: second chunk runs");
        check(ids_equal(part1, full_ids, 2 * ple::kNgramHeads),
              "split: first chunk equals full prefill");
        check(ids_equal(part2, full_ids + 2 * ple::kNgramHeads, 2 * ple::kNgramHeads),
              "split: second chunk equals full prefill");
    }
    {
        std::uint32_t context[2] = {11, 22};
        std::uint64_t chunk[3 * ple::kNgramHeads];
        check(ple::compute_ngram_ids(tables, context, 2, full_input, 3, kEos, chunk).ok(),
              "decode: chunk runs");
        std::uint32_t hist[5] = {11, 22, 5, 6, 7};
        ple::advance_ngram_context(hist, 5, context);
        const std::uint32_t next[1] = {8};
        std::uint64_t step[ple::kNgramHeads];
        check(ple::compute_ngram_ids(tables, context, 2, next, 1, kEos, step).ok(),
              "decode: single token runs");
        check(ids_equal(step, full_ids + 3 * ple::kNgramHeads, ple::kNgramHeads),
              "decode: one-token append equals full prefill");
    }
}

static void test_error_paths() {
    const ple::NgramIdTables tables = slot0_tables();
    const std::uint32_t context[3] = {1, 2, 3};
    const std::uint32_t input[1] = {4};
    std::uint64_t ids[ple::kNgramHeads];
    check(!ple::compute_ngram_ids(tables, context, 3, input, 1, kEos, ids).ok(),
          "err: context_len 3 rejected");
    check(!ple::compute_ngram_ids(tables, context, 2, nullptr, 1, kEos, ids).ok(),
          "err: null input rejected");
    check(!ple::compute_ngram_ids(tables, context, 2, input, 1, kEos, nullptr).ok(),
          "err: null output rejected");
    check(!ple::compute_ngram_ids(tables, nullptr, 2, input, 1, kEos, ids).ok(),
          "err: null context rejected");
    check(ple::compute_ngram_ids(tables, nullptr, 0, nullptr, 0, kEos, nullptr).ok(),
          "err: zero-length accepted");

    ple::NgramIdTables bad = tables;
    bad.vocab_sizes[0] = 0;
    check(!ple::compute_ngram_ids(bad, context, 2, input, 1, kEos, ids).ok(),
          "err: zero vocab size rejected");
    bad = tables;
    bad.multipliers[0] = 0;
    check(!ple::compute_ngram_ids(bad, context, 2, input, 1, kEos, ids).ok(),
          "err: zero multiplier rejected");

    ple::NgramHashParams params;
    params.unigram_vocab_size = 0;
    std::int64_t mult[ple::kNgramSize];
    check(!ple::build_layer_multipliers(params, mult).ok(),
          "err: zero unigram vocab rejected");

    std::int64_t sizes[ple::kNgramHeads];
    std::int64_t offsets[ple::kNgramHeads];
    for (std::uint32_t h = 0; h < ple::kNgramHeads; ++h) sizes[h] = 1;
    sizes[3] = -1;
    check(!ple::build_ngram_head_offsets(sizes, offsets).ok(),
          "err: negative size rejected");
}

int main() {
    printf("=== PLE n-gram hash (slot0/slot1 tables, EOS shift, ids) ===\n");
    test_splitmix64_vectors();
    test_tables_slot0();
    test_tables_slot1();
    test_shift_golden();
    test_ids_golden_cases();
    test_random_cases();
    test_prefill_split_and_decode();
    test_error_paths();
    printf("\n%d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
