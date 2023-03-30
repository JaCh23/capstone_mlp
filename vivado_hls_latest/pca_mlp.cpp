#include <math.h>
#include <cmath>
#include "hls_stream.h"
#include <ap_fixed.h>


#define NUMBER_OF_INPUT_WORDS 125  // length of an input vector
#define NUMBER_OF_OUTPUT_WORDS 3  // length of an output vector

struct AXIS_wLAST{
	float data;
	bool last;
};

//typedef ap_fixed<16, 15> data_t;
//typedef ap_fixed<16, 15> weight_t;

typedef float data_t;

const int N_INPUTS = 10;
const int N_HIDDEN_1 = 20;
const int N_HIDDEN_2 = 20;
const int N_OUTPUTS = 3;


const data_t mean[40*3] = {0.0010627939607925, -0.0003524002597669, 0.0002437739546460, 0.0031796998944501, -0.0010377653622596, 0.0006798579234687, 0.0063282730094288, -0.0020146526114660, 0.0012071150960644, 0.0104632190581124, -0.0032151359155494, 0.0016785696695427, 0.0155057020055698, -0.0045398375441890, 0.0019078916508091, 0.0213336091493404, -0.0058539979840261, 0.0016789958192061, 0.0277755115779176, -0.0069862278697234, 0.0007588475259049, 0.0346105527695270, -0.0077310917333278, -0.0010871187007938, 0.0415753467231003, -0.0078561626641846, -0.0040779451387370, 0.0483778423944215, -0.0071133034424773, -0.0084001241581087, 0.0547168501212796, -0.0052532095739009, -0.0141922445294796, 0.0603048226776459, -0.0020417354023642, -0.0215329515011496, 0.0648911181126343, 0.0027237771849548, -0.0304333461164088, 0.0682826378748814, 0.0091999294807625, -0.0408344149684493, 0.0703594569529958, 0.0174857782987900, -0.0526095199737445, 0.0710836885654377, 0.0276153203330474, -0.0655714564454707, 0.0705009024169590, 0.0395542275565838, -0.0794832009796152, 0.0687343821128180, 0.0532006413217848, -0.0940712444019442, 0.0659733502136349, 0.0683894592598885, -0.1090403297524080, 0.0624568256813010, 0.0848994288150679, -0.1240884310099533, 0.0584550364375895, 0.1024623691414395, -0.1389209285884512, 0.0542502741062556, 0.1207739356955962, -0.1532631142289425, 0.0501188204786702, 0.1395054855488851, -0.1668703928539642, 0.0463152489342533, 0.1583166894130940, -0.1795358140272314, 0.0430599268874681, 0.1768685541847026, -0.1910948543974184, 0.0405301303493480, 0.1948364556074400, -0.2014276068952580, 0.0388548398202643, 0.2119226952100137, -0.2104587011282906, 0.0381130294362795, 0.2278679553320020, -0.2181554252888449, 0.0383349911889938, 0.2424609439281820, -0.2245244289946610, 0.0395062503323305, 0.2555454286846714, -0.2296073678941732, 0.0415734702925145, 0.2670239727019336, -0.2334757842044266, 0.0444519006465795, 0.2768577645402706, -0.2362252756263228, 0.0480338177123157, 0.2850622680108888, -0.2379689844572598, 0.0521975619624208, 0.2916987525286755, -0.2388304881281170, 0.0568167339351380, 0.2968622099485292, -0.2389362624612487, 0.0617692372888037, 0.3006666502040594, -0.2384077679733488, 0.0669457188562656, 0.3032291305047745, -0.2373537457485308, 0.0722569719862679, 0.3046540531507962, -0.2358630443319590, 0.0776399495470940, 0.3050195468472357, -0.2339986023779794, 0.0830620560285755, 0.3043673477731616, -0.2317930468224238};

const data_t variance[40*3] = {0.0000005384699055, 0.0000002832539044, 0.0000011570241687, 0.0000047155967816, 0.0000025172797746, 0.0000102259192756, 0.0000181135187596, 0.0000098917009271, 0.0000398228303761, 0.0000477156804629, 0.0000268883881908, 0.0001068224754433, 0.0001006397032844, 0.0000590583155951, 0.0002303453671229, 0.0001830236054405, 0.0001128805028803, 0.0004296541092896, 0.0002993856145990, 0.0001957244839497, 0.0007222123184269, 0.0004526740988170, 0.0003158734433104, 0.0011220758290041, 0.0006450278379628, 0.0004825298865082, 0.0016387081901732, 0.0008790350349153, 0.0007057464135360, 0.0022763142067931, 0.0011590457907636, 0.0009962933226265, 0.0030337719607908, 0.0014919892753985, 0.0013655640555026, 0.0039052458934124, 0.0018872254654869, 0.0018256638360949, 0.0048815073914382, 0.0023552506656671, 0.0023897916422772, 0.0059518770492076, 0.0029054822708715, 0.0030729055823946, 0.0071065550561824, 0.0035437026898219, 0.0038924799431639, 0.0083390231752753, 0.0042699158527717, 0.0048690291928536, 0.0096481230764242, 0.0050772775083990, 0.0060260408450604, 0.0110394594228170, 0.0059524457576776, 0.0073890491094602, 0.0125258839853802, 0.0068772935269642, 0.0089838155121103, 0.0141269726257049, 0.0078315816786614, 0.0108338261066319, 0.0158675608614996, 0.0087959974330993, 0.0129575048192475, 0.0177755419812318, 0.0097549534217021, 0.0153655990291828, 0.0198792057375905, 0.0106986609472499, 0.0180590884697811, 0.0222044047946504, 0.0116241905016093, 0.0210277866436764, 0.0247717996655874, 0.0125354354352269, 0.0242496232280335, 0.0275943572877180, 0.0134420552149272, 0.0276904962896022, 0.0306752463973302, 0.0143576367905759, 0.0313045775990545, 0.0340062285232866, 0.0152973825507352, 0.0350350511838050, 0.0375666274507478, 0.0162757082307802, 0.0388153338695845, 0.0413229217380333, 0.0173041163345317, 0.0425708574916587, 0.0452290625888637, 0.0183896681281039, 0.0462214349395450, 0.0492275300486281, 0.0195343251648369, 0.0496841550371501, 0.0532511769823386, 0.0207353117563540, 0.0528766939445653, 0.0572259457453895, 0.0219865615242961, 0.0557209447897155, 0.0610745344983741, 0.0232811790118362, 0.0581469121746776, 0.0647209658571116, 0.0246147056057427, 0.0600968347839780, 0.0680960999140216, 0.0259888151792240, 0.0615294168899953, 0.0711435076973092, 0.0274148732390272, 0.0624238704831456, 0.0738252869871674, 0.0289166610323405, 0.0627832751438283, 0.0761268282443244};

const data_t pca_eigvecs[5*120] = {0.0096018156552972, -0.0726621247751870, 0.1225813717759564, 0.0083984890215394, -0.0722237952960019, 0.1224535733365655, 0.0065623223808443, -0.0715549058096963, 0.1222099956265975, 0.0040579866186016, -0.0706535365697900, 0.1217979008497689, 0.0008425160145626, -0.0695324273599707, 0.1211604730290984, -0.0031283780983116, -0.0682285964992220, 0.1202474892512609, -0.0078902916581530, -0.0668091162194804, 0.1190237157752685, -0.0134566349850602, -0.0653717808489826, 0.1174756857130925, -0.0198008333439334, -0.0640382156478962, 0.1156152512284155, -0.0268376879108119, -0.0629410173329538, 0.1134798219369365, -0.0344111358599033, -0.0622078750893863, 0.1111294615718775, -0.0422973682339186, -0.0619462934435794, 0.1086419190163978, -0.0502288678792090, -0.0622314304049021, 0.1061066315354015, -0.0579352847886790, -0.0630987523702636, 0.1036185104381791, -0.0651869766601199, -0.0645417116589435, 0.1012721238281496, -0.0718243194558070, -0.0665142173943607, 0.0991565440847895, -0.0777642982837884, -0.0689372620808668, 0.0973506685028393, -0.0829878152962837, -0.0717088175297334, 0.0959190232747158, -0.0875181472960945, -0.0747153920748415, 0.0949079732892381, -0.0914002458848200, -0.0778432576065071, 0.0943425816651446, -0.0946859442291114, -0.0809873874877121, 0.0942245075007750, -0.0974258340273256, -0.0840570033342750, 0.0945314367446354, -0.0996663474009510, -0.0869777710211553, 0.0952184597236465, -0.1014500452324049, -0.0896916523397893, 0.0962214860666319, -0.1028174702360652, -0.0921557112492705, 0.0974624551980553, -0.1038094883616679, -0.0943408337165558, 0.0988557197207830, -0.1044693804328948, -0.0962308321766612, 0.1003147823296046, -0.1048442697092083, -0.0978218188134855, 0.1017585347295969, -0.1049855143522792, -0.0991215343690287, 0.1031163664845880, -0.1049477277040825, -0.1001483017362197, 0.1043316613115719, -0.1047864709698910, -0.1009294331197104, 0.1053635767467194, -0.1045549550331283, -0.1014991129227583, 0.1061871435005930, -0.1043002533610857, -0.1018959457369574, 0.1067918800527296, -0.1040597386530170, -0.1021603098618380, 0.1071791773859132, -0.1038583354801709, -0.1023316405177831, 0.1073587449267976, -0.1037066958779219, -0.1024456513901017, 0.1073444567318981, -0.1036003064979386, -0.1025314734487301, 0.1071499874171126, -0.1035195104142571, -0.1026086543902812, 0.1067846768478974, -0.1034304367849324, -0.1026841455796275, 0.1062501069883626, -0.1032872428469715, -0.1027493669996406, 0.1055378811073523, -0.0857171307569434, 0.0276658033964055, -0.0566221715124538, -0.0879641625160951, 0.0283020716524773, -0.0582131643016010, -0.0912628727064041, 0.0291160561007370, -0.0605362471827210, -0.0955202560732934, 0.0299476572273382, -0.0635186758419815, -0.1006044618515672, 0.0305946723198444, -0.0670686142708662, -0.1063353693572691, 0.0308283669974698, -0.0710795176489200, -0.1124741227152507, 0.0304136138511631, -0.0754355138960858, -0.1187150619399410, 0.0291323089363527, -0.0800185466019044, -0.1246857682002814, 0.0268055893907001, -0.0847140542782673, -0.1299643790588204, 0.0233113877431325, -0.0894156313029855, -0.1341218308615233, 0.0185947398445503, -0.0940277921057692, -0.1367882401542539, 0.0126712051905612, -0.0984663049786410, -0.1377270774692853, 0.0056253874171353, -0.1026565261194416, -0.1368884085498319, -0.0023933147955500, -0.1065306493772996, -0.1344156378467476, -0.0111775089238583, -0.1100250587070820, -0.1306023471559245, -0.0204724521688090, -0.1130789172965765, -0.1258202552703935, -0.0299921119225984, -0.1156350420466974, -0.1204481829061735, -0.0394409793514590, -0.1176434677021849, -0.1148224910666939, -0.0485392490886648, -0.1190674325689982, -0.1092137282189240, -0.0570465790166197, -0.1198905554474836, -0.1038233777101066, -0.0647792366377481, -0.1201233004587866, -0.0987916410893195, -0.0716173893263965, -0.1198065924330124, -0.0942089614161878, -0.0775025395441224, -0.1190109987526539, -0.0901271139664523, -0.0824279171955525, -0.1178311545548062, -0.0865682458452177, -0.0864257594614169, -0.1163764822141104, -0.0835317035584377, -0.0895548677435171, -0.1147602540225836, -0.0809993256580724, -0.0918904064794767, -0.1130892486442000, -0.0789397602035649, -0.0935165059695059, -0.1114557597453601, -0.0773122980463628, -0.0945213160887048, -0.1099328101095100, -0.0760704942230187, -0.0949937859404827, -0.1085726304700944, -0.0751655169173298, -0.0950214529250011, -0.1074077673965876, -0.0745489183163307, -0.0946887177833245, -0.1064539802781129, -0.0741746653959784, -0.0940752455686265, -0.1057140368000324, -0.0740001733402821, -0.0932543169146105, -0.1051816077137958, -0.0739863271794852, -0.0922909479370835, -0.1048446792913572, -0.0740965388463440, -0.0912396948074439, -0.1046881268372945, -0.0742952799473261, -0.0901421137492290, -0.1046952910195304, -0.0745464148651735, -0.0890239616565076, -0.1048486063394150, -0.0748119072468176, -0.0878923060651030, -0.1051294443203099, -0.0750515132295455, -0.0867328674963775, -0.1055174711482865, -0.0051683071145011, 0.1294294745343782, 0.0456320642897839, -0.0062776317660478, 0.1322092243001515, 0.0460797901915309, -0.0079433436199651, 0.1361982488595909, 0.0466843043227585, -0.0101619247097812, 0.1411791350657162, 0.0473718975678053, -0.0129287708125539, 0.1468706240111756, 0.0480525675773056, -0.0162350561754520, 0.1529410707051810, 0.0486263535841774, -0.0200612037513778, 0.1590327273874357, 0.0489921405333132, -0.0243708580433295, 0.1647913159568022, 0.0490554388502993, -0.0291036049533078, 0.1698960136521681, 0.0487358284211356, -0.0341682187247152, 0.1740821143299188, 0.0479729481786240, -0.0394419346499415, 0.1771516686139184, 0.0467301258970840, -0.0447796542623263, 0.1789724890093627, 0.0449960104374107, -0.0500340253306967, 0.1794703040911373, 0.0427847420365124, -0.0550807206231483, 0.1786201373293601, 0.0401351848049056, -0.0598391408269938, 0.1764415225556318, 0.0371097482473842, -0.0642794982335322, 0.1729985365368944, 0.0337931152436451, -0.0684150832706450, 0.1684021884232639, 0.0302907122470850, -0.0722854823025027, 0.1628105866578070, 0.0267262690735226, -0.0759387590419742, 0.1564227199413967, 0.0232375643505229, -0.0794179650509077, 0.1494643927682494, 0.0199696016070657, -0.0827537234549604, 0.1421685644843313, 0.0170650009583419, -0.0859619399787146, 0.1347548426066150, 0.0146523918376222, -0.0890447687923642, 0.1274131242229772, 0.0128345283564281, -0.0919931346887517, 0.1202943793182752, 0.0116783650215733, -0.0947897874598552, 0.1135088845271822, 0.0112091148627381, -0.0974124192737945, 0.1071302270834953, 0.0114093623941626, -0.0998366746450079, 0.1012025142996562, 0.0122231569024059, -0.1020391470763967, 0.0957485068339381, 0.0135639257593679, -0.1040001480482892, 0.0907770679577287, 0.0153245197044785, -0.1057060038714850, 0.0862891050701690, 0.0173876788892709, -0.1071503603779532, 0.0822817962386153, 0.0196355179074942, -0.1083341414221634, 0.0787513084954811, 0.0219571023159862, -0.1092640732471989, 0.0756942714753928, 0.0242538043872119, -0.1099499910331626, 0.0731083343877068, 0.0264424807680521, -0.1104012713854568, 0.0709920801567684, 0.0284567325297097, -0.1106231570331544, 0.0693444616514028, 0.0302466631282485, -0.1106133632122775, 0.0681638539980988, 0.0317777297951258, -0.1103596026677723, 0.0674468362572909, 0.0330291106408546, -0.1098389188110621, 0.0671866100960978, 0.0339919987927390, -0.1090193375223213, 0.0673710944087781, 0.0346682024086842, -0.2591657383224724, 0.1390253082986996, -0.0431752757530482, -0.2575256312527808, 0.1345035041842782, -0.0412983179940133, -0.2547619087782884, 0.1275126545377909, -0.0384471397550786, -0.2505323554378056, 0.1178750351125695, -0.0345909831358618, -0.2444137334026454, 0.1054667018975237, -0.0297084482100167, -0.2359350260962210, 0.0902915340391453, -0.0237988595639962, -0.2246236806632044, 0.0725429106420080, -0.0168925284493457, -0.2100701195462400, 0.0526374466725920, -0.0090585308121913, -0.1920143591049744, 0.0312102924411426, -0.0004078372290061, -0.1704493761085122, 0.0090722928999055, 0.0089080691605172, -0.1457181577832787, -0.0128587528730875, 0.0187015097386130, -0.1185630813106417, -0.0336358231229342, 0.0287553170159067, -0.0900834248797075, -0.0523565168529309, 0.0388306991970799, -0.0615865818800789, -0.0682268518445978, 0.0486754380090233, -0.0343725202877394, -0.0806149321079325, 0.0580323437096278, -0.0095289398784399, -0.0890952579413109, 0.0666485326467041, 0.0121941927715480, -0.0934806793110108, 0.0742861245255782, 0.0304094019236119, -0.0938348390092211, 0.0807346596819449, 0.0450399752212096, -0.0904579481570218, 0.0858249408852414, 0.0562338289286551, -0.0838437410348196, 0.0894427445002449, 0.0642774262115040, -0.0746140509653360, 0.0915399858339940, 0.0695274818605473, -0.0634446527763388, 0.0921405617486388, 0.0723647913550316, -0.0509975703708240, 0.0913388374970758, 0.0731673742429368, -0.0378707987668352, 0.0892904303802523, 0.0722977695519760, -0.0245694307783072, 0.0861968227807803, 0.0700992435905552, -0.0114960343678085, 0.0822868068784400, 0.0668963865579361, 0.0010451634404516, 0.0777979777868366, 0.0629962312401672, 0.0128362194680021, 0.0729609129201106, 0.0586874420439767, 0.0237303612453594, 0.0679873917212169, 0.0542363789060872, 0.0336368057320255, 0.0630629374111550, 0.0498803376338225, 0.0425071531969130, 0.0583431996519772, 0.0458198826164643, 0.0503239885463586, 0.0539533369707433, 0.0422126615942462, 0.0570916811781820, 0.0499894436400704, 0.0391707983145484, 0.0628290309258996, 0.0465212665733558, 0.0367625142412064, 0.0675632775840262, 0.0435956744394631, 0.0350175628906258, 0.0713249404081526, 0.0412402311585667, 0.0339347991989362, 0.0741431248443802, 0.0394665184317530, 0.0334901613143907, 0.0760412236556894, 0.0382730319664459, 0.0336431161262458, 0.0770331642170334, 0.0376471815228652, 0.0343409544777520, 0.0771206647977966, 0.0375662158351584, 0.0008281712803276, -0.1310230954563874, -0.1546434641727868, -0.0001234999290328, -0.1332655397617519, -0.1548823162748459, -0.0015841939646604, -0.1362797267734586, -0.1550706506953146, -0.0035819229822596, -0.1396582381527163, -0.1550158082304734, -0.0061473666992332, -0.1428776971064152, -0.1544904771020926, -0.0093069033805803, -0.1453377304219674, -0.1532454018580620, -0.0130742256709025, -0.1464103042404045, -0.1510318607540374, -0.0174387528533164, -0.1455018403728998, -0.1476186299590863, -0.0223533343648960, -0.1421153841557439, -0.1428085128347490, -0.0277228407295390, -0.1359006316518668, -0.1364495763273454, -0.0334005985474512, -0.1266846651198186, -0.1284427799201123, -0.0391975264844563, -0.1144828439489194, -0.1187478756126673, -0.0449064862913630, -0.0994951196454610, -0.1073880205687574, -0.0503357365210446, -0.0820931924923990, -0.0944548994845042, -0.0553395992108002, -0.0628006472996170, -0.0801140704638000, -0.0598338655888264, -0.0422634919004982, -0.0646090208565471, -0.0637924550589668, -0.0212070857649649, -0.0482616120379114, -0.0672302735596221, -0.0003787066691290, -0.0314652258335057, -0.0701817848801564, 0.0195189235728605, -0.0146676849012725, -0.0726827946818499, 0.0378915597331791, 0.0016572568356717, -0.0747589491535188, 0.0543011150885381, 0.0170478638709215, -0.0764209934468676, 0.0684879220381884, 0.0310961301326712, -0.0776654069070874, 0.0803662229424340, 0.0434853361223384, -0.0784791385476400, 0.0899987096632665, 0.0540156388801445, -0.0788469517801118, 0.0975594581240211, 0.0626127660325660, -0.0787605182417849, 0.1032942274156633, 0.0693194270280643, -0.0782277530675203, 0.1074843083557381, 0.0742732517940333, -0.0772806944079486, 0.1104171791458771, 0.0776773172601176, -0.0759801767517047, 0.1123647398921514, 0.0797695582720171, -0.0744159732079413, 0.1135687357062140, 0.0807960270954222, -0.0727016846511872, 0.1142326642673719, 0.0809905853648726, -0.0709652846585141, 0.1145189015968247, 0.0805617684482261, -0.0693373023854980, 0.1145499293699899, 0.0796859845067996, -0.0679389387867352, 0.1144124127078308, 0.0785056686988234, -0.0668721130463227, 0.1141628295017032, 0.0771304706575816, -0.0662128262936465, 0.1138335205471112, 0.0756400371574290, -0.0660074449203099, 0.1134380258463010, 0.0740874707423957, -0.0662713486534796, 0.1129751788244980, 0.0725024765715495, -0.0669889919367205, 0.1124314646841640, 0.0708941462099047, -0.0681150288317925, 0.1117818324493728, 0.0692531244573005};

const data_t W1[10][20] = {
	{-0.4664726555347443, -0.2932562828063965, -0.1790020316839218, -0.0928153023123741, -0.2155409604310990, 0.0221045780926943, -0.2371978461742401, 0.3890635371208191, 0.2627318799495697, -0.0956153050065041, 0.4823631644248962, -0.2227180451154709, -0.0458677485585213, 0.0413214601576328, 0.2694519460201264, -0.0279156304895878, -0.3371381163597107, -0.0728957653045654, -0.1622941344976425, 0.0080246068537235},
	{0.4334985315799713, -0.2253865152597428, 0.1707311719655990, 0.1244070678949356, -0.2179765999317169, -0.3142164945602417, -0.4137811362743378, 0.2490384727716446, -0.5076500773429871, 0.0628167018294334, -0.2641243338584900, -0.0686718076467514, 0.2895467877388000, -0.2597183585166931, -0.1606632620096207, -0.4153786599636078, -0.3449614644050598, 0.4031424820423126, 0.5171442031860352, 0.0619519874453545},
	{-0.1676063090562820, 0.2605653405189514, -0.1880410164594650, -0.0342629179358482, 0.1949465125799179, -0.3713461458683014, -0.0968480706214905, -0.0642694905400276, 0.3270575404167176, 0.4256795346736908, 0.3386985957622528, -0.5536661744117737, 0.1368671953678131, 0.2441957294940948, -0.3149576783180237, 0.1735195219516754, -0.3933201134204864, -0.2850896716117859, 0.2374391704797745, 0.3423698544502258},
	{0.0920678079128265, 0.0781316459178924, 0.2225085645914078, 0.3994671702384948, -0.3378432989120484, 0.4487757682800293, -0.0232328884303570, 0.0160473734140396, 0.3409566283226013, -0.4212338924407959, -0.2543828487396240, 0.2345660924911499, -0.0275482032448053, -0.4038698375225067, 0.2034331262111664, 0.0481856949627399, -0.3833925127983094, -0.1858133673667908, 0.3327524065971374, -0.2446399331092834},
	{0.0451963469386101, 0.7405304312705994, 0.1834186613559723, -0.1085036098957062, 0.2836367785930634, -0.0114038335159421, 0.1900068372488022, -0.3461838364601136, 0.5346325635910034, 0.3522671461105346, -0.0776355862617493, 0.6513167619705200, -0.3571615815162658, -0.0607369616627693, 0.4744910895824432, 0.1715861707925796, -0.1983971446752548, 0.2325906306505203, -0.2745390236377716, -0.2240966856479645},
	{0.1557404845952988, 0.4567954242229462, 0.5452381372451782, 0.0527549721300602, -0.3143732845783234, -0.0016845612553880, 0.1267316639423370, -0.4347758889198303, 0.2582679986953736, 0.3901956677436828, 0.3980911076068878, 0.5663527846336365, 0.2811873853206634, -0.0574794486165047, 0.3550390303134918, 0.3644282519817352, -0.1784973442554474, -0.3387299180030823, 0.0993083342909813, -0.3286721110343933},
	{0.3438642323017120, -0.0147827789187431, 0.5948360562324524, 0.4257244169712066, -0.0023023006506264, 0.2792742550373078, 0.1270595341920853, -0.3060417175292969, -0.0851726979017258, -0.2331092953681946, -0.2985285520553589, -0.3556640446186066, -0.0258837174624205, -0.2686784863471985, -0.2933332920074463, -0.3967798352241516, -0.0461033396422863, -0.1544699370861054, 0.1285713315010071, -0.1032934859395027},
	{-0.4147362709045410, -0.0028223777189851, -0.4438364803791046, 0.0144399357959628, 0.3917037844657898, 0.2207552790641785, -0.4941808581352234, 0.4197439849376678, -0.1377366185188294, -0.3283815681934356, -0.5866543054580688, -0.1606201380491257, 0.0731147304177284, -0.3427023589611054, 0.2108420133590698, -0.3565674722194672, -0.4007642269134522, -0.1895882636308670, -0.6256232857704163, -0.1050762981176376},
	{0.1207763701677322, -0.4768770635128021, 0.2228737324476242, 0.0789346024394035, -0.1697681993246078, -0.1626687496900558, -0.3158157467842102, -0.4733255505561829, 0.4175893664360046, -0.2454728335142136, 0.0162324476987124, -0.0075450385920703, 0.4592466056346893, 0.1539337635040283, -0.1389209032058716, 0.4200795590877533, 0.4331639409065246, 0.0051548676565290, 0.5737369656562805, 0.5424619317054749},
	{0.2512264847755432, -0.3196353912353516, 0.4166171550750732, 0.1696787774562836, 0.3588902354240418, -0.1137296929955482, 0.2483225911855698, -0.1960584223270416, 0.0799910426139832, -0.1984701752662659, 0.4018895030021668, 0.4407009184360504, 0.2093299329280853, 0.0053143925033510, 0.4371486008167267, 0.3130356967449188, -0.2058875411748886, 0.2443268895149231, -0.0477242767810822, 0.1066340208053589}
};

const data_t B1[20] = {0.1078734174370766, -0.1471994221210480, 0.1491281837224960, 0.0756862312555313, -0.2175213247537613, 0.0320225581526756, -0.1276378780603409, -0.0448067151010036, 0.0124999806284904, 0.0053237169049680, 0.1844532489776611, 0.0318245626986027, 0.0861481577157974, 0.2510425746440888, 0.0547797046601772, -0.0039370176382363, 0.0467316173017025, 0.1073130667209625, 0.1588254272937775, 0.2809619307518006};

const data_t W2[20][20] = {
	{-0.2407088875770569, 0.4102632105350494, 0.2082888185977936, 0.1329980343580246, -0.2803157567977906, 0.1299425512552261, 0.3446008265018463, -0.3209764659404754, 0.0878489762544632, 0.2241237759590149, -0.2917254269123078, -0.3267565071582794, 0.1341335028409958, 0.1429931074380875, 0.3078923821449280, -0.1704458147287369, 0.1349942684173584, -0.0576391629874706, 0.1238452196121216, -0.2256617993116379},
	{0.5220613479614258, -0.2774812579154968, -0.0125123225152493, 0.1608531028032303, -0.3385921418666840, -0.0195270199328661, -0.1138810440897942, -0.0398379787802696, -0.2918057441711426, -0.0266861412674189, 0.2726963162422180, 0.0144524276256561, -0.3621947467327118, -0.3638000488281250, 0.3694520890712738, 0.1071900948882103, -0.3950257003307342, 0.1467426568269730, -0.0645999163389206, 0.2348843663930893},
	{0.5207722187042236, 0.0159450490027666, -0.0306785963475704, -0.0914760231971741, -0.6040324568748474, 0.0688011869788170, -0.1448137164115906, -0.0602239221334457, 0.0041642133146524, -0.1267061829566956, 0.0042419917881489, -0.2604089677333832, -0.1655176132917404, 0.1930093467235565, 0.0932118669152260, 0.7110210061073303, 0.4907216131687164, -0.3855744898319244, -0.3440333306789398, -0.4334673881530762},
	{0.1887234151363373, -0.2114804834127426, 0.1103106066584587, 0.1483763605356216, 0.0290003325790167, -0.3213957250118256, -0.1299182325601578, -0.1030841767787933, 0.3545232117176056, 0.2382456958293915, -0.2280456125736236, 0.3176781833171844, 0.3511204719543457, 0.1136437729001045, 0.1291459947824478, -0.0419037342071533, 0.0297397859394550, 0.3051060140132904, -0.0752846598625183, 0.0560967698693275},
	{0.0447985529899597, -0.0258477684110403, -0.4481566548347473, -0.0156316589564085, -0.2396641820669174, 0.0083512533456087, -0.3633345365524292, -0.3144897520542145, -0.2836622297763824, -0.1060631424188614, 0.1360859721899032, -0.3416938185691834, 0.1286702752113342, -0.0860216245055199, 0.2284649610519409, 0.5367923974990845, -0.0581068284809589, -0.1262946724891662, -0.2408636957406998, 0.1298396438360214},
	{0.3220036327838898, 0.2671949267387390, 0.4531162679195404, -0.3068740963935852, 0.3493172526359558, 0.1299442946910858, -0.2280968874692917, 0.2450632303953171, 0.0203769914805889, 0.1038151755928993, -0.2065529227256775, -0.4144398868083954, 0.0719060674309730, -0.0977573394775391, 0.3013325631618500, 0.4078135490417480, -0.0303267464041710, 0.1460956782102585, -0.3545310199260712, 0.0793866813182831},
	{0.3345502316951752, 0.0869645178318024, 0.3083061873912812, 0.0762804597616196, 0.3538979589939118, -0.1512416303157806, 0.0373030640184879, 0.0820361748337746, -0.0181934442371130, 0.2091014087200165, 0.2189135849475860, 0.1037597060203552, -0.0401911139488220, -0.0096167940646410, 0.3879909813404084, -0.0953954979777336, -0.1823589652776718, 0.1189848855137825, -0.2861693203449250, -0.2970968782901764},
	{-0.1846610605716705, -0.2688624262809754, -0.0661104768514633, 0.1359891295433044, 0.4613337516784668, -0.2894588112831116, 0.5351104736328125, 0.4148063361644745, 0.0923863276839256, -0.0446501486003399, 0.2871250212192536, -0.1429143846035004, -0.3818635940551758, 0.1670626103878021, -0.2629080712795258, -0.5231230854988098, -0.1811424344778061, -0.0802385658025742, 0.1122328937053680, 0.1480911672115326},
	{0.0243812482804060, -0.3696094155311584, -0.0453531518578529, -0.3484322130680084, -0.1248790696263313, -0.3872357606887818, 0.2503264546394348, -0.1278537660837174, -0.0595497228205204, 0.2485809922218323, 0.0858180820941925, -0.2302842885255814, -0.1878259927034378, 0.2154788523912430, 0.3868639767169952, 0.4482441246509552, 0.1591666042804718, 0.2372088730335236, 0.1568889617919922, -0.2162039577960968},
	{0.4519921541213989, -0.1651847213506698, -0.3757913708686828, 0.1154499575495720, 0.1725660264492035, -0.1928283870220184, -0.2242643088102341, -0.1807581633329392, -0.1362761855125427, 0.2132647633552551, -0.2871508300304413, 0.3024182319641114, 0.1030984744429588, -0.1532859206199646, 0.0710856691002846, 0.1968313455581665, -0.0814498066902161, -0.1529562771320343, 0.3010145723819732, 0.0730115324258804},
	{0.2731848359107971, 0.3746819794178009, -0.0400747954845428, -0.0358279049396515, 0.2938065826892853, 0.2758993506431580, 0.3075035512447357, -0.1193391084671020, -0.0141050182282925, -0.1756028085947037, 0.3109324574470520, 0.0128899253904819, -0.4823915362358093, -0.0610831007361412, 0.1618936806917190, -0.0570095777511597, -0.0606650263071060, 0.4952582120895386, 0.4808960258960724, -0.1210802495479584},
	{0.3833227157592774, 0.2021435052156448, 0.2822023034095764, -0.0245467964559793, -0.5814124345779419, 0.0032548401504755, 0.0079144556075335, 0.1576438248157501, 0.3484385907649994, 0.0301770307123661, -0.2076360881328583, -0.1361643970012665, 0.1887126117944718, -0.2309578508138656, 0.2966031134128570, 0.7353038787841797, -0.3489756286144256, 0.3581102192401886, -0.4206749498844146, 0.0366363339126110},
	{-0.5229215621948242, 0.1002913489937782, -0.0502535253763199, -0.2582831382751465, 0.3941234052181244, -0.2716005742549896, -0.2881867289543152, 0.2013061791658402, 0.1758260577917099, -0.3616335690021515, -0.1307666301727295, -0.1178024187684059, 0.1043831333518028, -0.0771276727318764, 0.1624925136566162, -0.6147513985633850, 0.3513834476470948, -0.0914529785513878, 0.2973491251468658, 0.0191439241170883},
	{-0.0820676535367966, 0.1650553941726684, -0.1016915068030357, 0.0191343538463116, -0.0637163966894150, -0.3700836598873138, -0.1054585874080658, 0.0443775393068790, -0.1767053604125976, -0.0989208519458771, 0.2615805566310882, -0.2188534736633301, -0.0593784265220165, 0.3292052745819092, -0.2344781607389450, -0.0064908917993307, 0.2458884418010712, 0.3770986795425415, -0.1238315254449844, 0.2780212759971618},
	{0.0703726410865784, -0.3779152929782868, 0.2568414509296417, -0.3631826341152191, -0.2641414403915406, -0.1912304759025574, -0.1240955144166946, 0.2014391273260116, 0.1277517676353454, -0.3180125653743744, 0.2662308514118194, -0.2104236185550690, -0.0357376448810101, -0.1618701666593552, 0.0552589409053326, 0.5609405636787415, -0.3122920393943786, 0.1997144222259522, 0.3322972059249878, 0.1990182399749756},
	{0.1206529438495636, 0.3252823352813720, 0.1535157561302185, -0.0221266150474548, 0.0742161050438881, -0.2584941089153290, 0.2496218085289002, -0.1302119791507721, 0.0834823846817017, 0.2637729644775390, 0.2397567331790924, 0.0670562461018562, 0.1439567655324936, -0.1353159546852112, 0.0114116575568914, 0.0016515089664608, -0.0148387169465423, -0.0085466345772147, -0.0375762693583965, 0.0759961530566216},
	{-0.0532546900212765, -0.2895802855491638, 0.2072826027870178, -0.0701278001070023, 0.5770543813705444, 0.0582046136260033, -0.1974323540925980, -0.4430475234985352, 0.2919197976589203, -0.0928468853235245, 0.0886028930544853, -0.2357773631811142, -0.1147908419370651, 0.0784793049097061, 0.0776728689670563, -0.1016127094626427, 0.2261238247156143, 0.1969847679138184, 0.1309296637773514, -0.0812152251601219},
	{-0.2284002453088760, 0.3959896266460418, 0.0503505803644657, -0.1186987981200218, 0.1440354138612747, -0.2576364576816559, 0.2462940514087677, -0.0591194443404674, 0.3208372592926026, 0.0453682728111744, 0.0229491349309683, 0.1836877912282944, 0.4259971380233764, -0.3036952614784240, -0.1671846508979797, -0.3295957446098328, 0.0932840108871460, -0.1515146493911743, 0.0649420022964478, 0.1125114560127258},
	{-0.1628368943929672, 0.1514585167169571, -0.1525449305772781, -0.1369895935058594, -0.1674629449844360, 0.1301631629467010, -0.0566997677087784, 0.1190654858946800, 0.3675479888916016, -0.1936141550540924, 0.0381543003022671, 0.2899520397186280, 0.5447025299072266, -0.3324365615844726, 0.0602468363940716, -0.1181203275918960, 0.0393932200968266, -0.1512912362813950, -0.1240832656621933, -0.1171622052788734},
	{-0.1023881584405899, 0.1830941736698151, 0.0362046547234058, 0.0006536692380905, 0.3321869075298310, 0.1614937186241150, -0.0893163084983826, -0.2141946256160736, 0.0005053135682829, 0.1632454097270966, 0.4503962099552155, 0.3301391899585724, 0.2018927931785584, 0.4500492513179779, -0.2189846187829971, 0.0492535345256329, 0.2224648445844650, 0.3621256649494171, -0.0484002567827702, 0.2574959695339203}
};

const data_t B2[20] = {-0.0166535079479218, 0.0717739388346672, -0.1050861552357674, -0.0176355801522732, 0.4450478553771972, -0.0451611690223217, -0.0294698234647512, -0.0628161206841469, 0.1319564282894134, 0.0683057084679604, 0.0919296815991402, -0.0644681528210640, 0.1585183590650558, 0.2335953563451767, -0.0520883277058601, -0.0832036584615707, 0.3445152342319488, 0.1750027537345886, 0.0603856481611729, 0.0620766580104828};

const data_t W3[20][3] = {
	{-0.3715395331382752, -0.3760049045085907, 0.1578935384750366},
	{0.2302922010421753, 0.4700850248336792, -0.6354692578315735},
	{-0.4147764444351196, -0.1365227848291397, -0.0690018683671951},
	{0.0063249208033085, -0.2328991740942002, -0.4393149912357330},
	{0.1167571172118187, -0.8101320266723633, -0.5214633345603943},
	{0.3802009224891662, -0.0932816118001938, -0.0210716780275106},
	{0.4821982979774475, 0.1772132366895676, -0.2133162766695022},
	{0.4678834676742554, -0.1004426479339600, -0.1299628168344498},
	{-0.3007581233978272, 0.2512927949428558, -0.0206308029592037},
	{0.4018850922584534, -0.4736098051071167, -0.2681105136871338},
	{0.3315300047397614, 0.0822849273681641, 0.0794309675693512},
	{0.0137617345899343, -0.0024417156819254, 0.0989088937640190},
	{0.0613554865121841, 0.6210366487503052, 0.1494643092155456},
	{0.7282825112342834, -0.0275648310780525, -0.6499463915824890},
	{-0.5818098187446594, 0.2810975015163422, 0.4215601682662964},
	{-0.5553271770477295, -0.7024034261703491, 0.0448515973985195},
	{0.2806389033794403, 0.5134183168411255, -0.5995533466339111},
	{0.2333417236804962, -0.5092228651046753, 0.0660022124648094},
	{0.6002216339111328, -0.4926484823226929, -0.5408765673637390},
	{0.4350499808788300, 0.2924855947494507, -0.3254205882549286}
};

const data_t B3[3] = {0.0481126345694065, 0.0201039332896471, -0.0642137750983238};

data_t relu(data_t x) {
    return x < 0 ? (data_t)0.0 : x;
}

data_t sigmoid(data_t x) {
    return 1.0 / (1.0 + std::exp(-x));
}

void softmax(data_t input[N_OUTPUTS], data_t output[N_OUTPUTS]) {
    data_t max_val = input[0];
    for (int i = 1; i < N_OUTPUTS; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    data_t sum = 0.0;
    for (int i = 0; i < N_OUTPUTS; i++) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    for (int i = 0; i < N_OUTPUTS; i++) {
        output[i] /= sum;
    }
}

void pca_mlp(hls::stream<AXIS_wLAST>& S_AXIS, hls::stream<AXIS_wLAST>& M_AXIS){
	#pragma HLS INTERFACE ap_ctrl_none port=return
	#pragma HLS INTERFACE axis port=S_AXIS
	#pragma HLS INTERFACE axis port=M_AXIS

    int word_cnt;
	AXIS_wLAST input_data, output_data;


	data_t input[125];
	// Read inputs from input stream
	for(int word_cnt = 0; word_cnt < NUMBER_OF_INPUT_WORDS; word_cnt++){
		input_data = S_AXIS.read();
		input[word_cnt] = input_data.data;
	}

	data_t action[40][3];
	    for (int i = 0; i < 40; i++) {
	        for (int j = 0; j < 3; j++) {
	            action[i][j] = input[i*3+j];
	        }
	    }

	    data_t scaled_action[40][3];
	    for (int i = 0; i < 40; i++) {
	        for (int j = 0; j < 3; j++) {
	            scaled_action[i][j] = (action[i][j] - mean[i*3+j]) / std::sqrt(variance[i*3+j]);
	        }
	    }

	    data_t scaled_action_vec[120];
	    int k = 0;
	    for (int i = 0; i < 40; i++) {
	        for (int j = 0; j < 3; j++) {
	            scaled_action_vec[k] = scaled_action[i][j];
	            k++;
	        }
	    }

	    // declare a 2D array with dimensions 120 x 5
	    data_t pca_eigvecs_2d[5][120];

	    // copy the data from the 1D array to the 2D array
	    for (int i = 0; i < 5; i++) {
	        for (int j = 0; j < 120; j++) {
	            pca_eigvecs_2d[i][j] = pca_eigvecs[120*i + j];
	        }
	    }

	    data_t pca_action[5];
	    for (int i = 0; i < 5; i++) {
	    	pca_action[i] = 0;
	        for (int j = 0; j < 120; j++) {
	        	pca_action[i] += scaled_action_vec[j] * pca_eigvecs_2d[i][j];
	        }
	    }


	    data_t mlp_input[10];
	    for (int i = 0; i < 5; i++) {
	        mlp_input[i] = pca_action[i];
	        mlp_input[i+5] = input[120+i];
	    }

    // Compute the output of the hidden layer
	data_t hidden1[N_HIDDEN_1];
	for (int i = 0; i < N_HIDDEN_1; i++) {
	     hidden1[i] = B1[i];
	     for (int j = 0; j < N_INPUTS; j++) {
	          hidden1[i] += mlp_input[j] * W1[j][i];
	        }
	        hidden1[i] = relu(hidden1[i]);
	    }

	    // Compute the output of the second hidden layer
	    data_t hidden2[N_HIDDEN_2];
	    for (int i = 0; i < N_HIDDEN_2; i++) {
	        hidden2[i] = B2[i];
	        for (int j = 0; j < N_HIDDEN_1; j++) {
	            hidden2[i] += hidden1[j] * W2[j][i];
	        }
	        hidden2[i] = relu(hidden2[i]);
	    }

	    // Compute the output of the output layer
	    data_t output_raw[N_OUTPUTS];
	    data_t output[N_OUTPUTS];
	    for (int i = 0; i < N_OUTPUTS; i++) {
	        output_raw[i] = B3[i];
	        for (int j = 0; j < N_HIDDEN_2; j++) {
	            output_raw[i] += hidden2[j] * W3[j][i];
	        }
	    }
	    softmax(output_raw, output);

    // Set the output data to the softmax array
    for(int i = 0; i < N_OUTPUTS; i++){
        output_data.data = output[i];
        output_data.last = (i == N_OUTPUTS-1) ? 1 : 0;
        M_AXIS.write(output_data);
    }
}
