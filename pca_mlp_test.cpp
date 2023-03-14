#include <stdio.h>
#include <math.h>
#include "hls_stream.h"

/***************** AXIS with TLAST structure declaration *********************/

struct AXIS_wLAST{
	float data;
	bool last;
};


/***************** Coprocessor function declaration *********************/

void pca_mlp(hls::stream<AXIS_wLAST>& S_AXIS, hls::stream<AXIS_wLAST>& M_AXIS);


/***************** Macros *********************/
#define NUMBER_OF_INPUT_WORDS 35  // length of an input vector
#define NUMBER_OF_OUTPUT_WORDS 4  // length of an input vector
#define NUMBER_OF_TEST_VECTORS 1  // number of such test vectors (cases)

/************************** Variable Definitions *****************************/
//float test_input_memory [NUMBER_OF_TEST_VECTORS*NUMBER_OF_INPUT_WORDS] = {0.017314464190467227f,0.7821381516657591f,1.0f,0.7851502881374393f,0.8214685624410034f,0.777309756018898f,1.0f,0.7438152161080576f,0.8758875965614796f,0.0f,0.8176854178020512f,0.6521847636448815f,0.9915502727104807f,0.14720052571656392f,0.2867605526436922f,0.0f,0.0f,1.0f,0.0f,0.8176854178020512f,0.6521847636448815f,1.0f,0.0f,0.0f,0.7327470560085914f,0.7670054540398284f,0.7876194904561352f,0.0003777085979821404f,0.004562143134149377f,0.015229133588794186f,0.008489068444362813f,0.0f,0.009446904812559165f,0.9824486506311336f,0.5290985126034317f,0.38614894284325796f,0.9494529308687075f,0.0f,0.07397361806125458f,0.59476694098463f,0.7198436209281258f,0.26609692233259596f,0.9824486506311336f,0.5290985126034317f,0.38614894284325796f,0.5751971490522738f,0.5788592538242026f,0.6546772353702576f,0.6303608153040674f,0.6459125042423195f,0.7726308539512028f,0.009468520442085682f,0.011712710762579626f,0.017824833108515348f,0.005093946644663561f,0.019465654191712645f,0.015376622627928382f,0.83442008939342f,0.41204243574476007f,0.9999999999999999f,0.9916250943994421f,0.3075444576465419f,0.3291439600701033f,0.5410450286608446f,0.665238985485304f,0.1066198430894188f,0.83442008939342f,0.41204243574476007f,0.9999999999999999f,0.5787903934852892f,0.6250082118276563f,0.5450281584503717f,0.6522639521708093f,0.5505606440131224f,0.7592430676622473f,0.0f,0.0f,0.0f,0.0f,0.0047103240218238085f,0.0f,0.7469269261405295f,0.6540769780819387f,0.48900901327472834f,1.0f,0.2661700890020115f,0.2057585278383944f,0.583041640151744f,0.7457313758113601f,0.09223203419799791f,0.7469269261405295f,0.6540769780819387f,0.48900901327472834f,0.577763837422617f,0.5693925573430506f,0.5870560088071526f,0.0f,0.7878431164406379f,0.6870106909630802f,0.1287997530906948f,0.04890807975145384f,0.11378937234541442f,0.07753320210333703f,0.3867255070794684f,0.12348765216436264f,0.8663248888827084f,1.0f,0.28347135356732167f,0.2654557598637883f,0.9525514116146542f,0.4973485965097738f,0.36541609268994457f,0.8711934524925733f,0.11531701824106748f,0.8663248888827084f,1.0f,0.28347135356732167f,0.5362222766191846f,0.4115857831263833f,0.8471261015945116}; // 4 inputs * 2
float test_input_memory [NUMBER_OF_TEST_VECTORS*NUMBER_OF_INPUT_WORDS] = {-9.20434773,-4.93421279,-0.7165668,-5.35652778,1.16597442,0.83953718,2.46925983,0.55131264,-0.1671036,0.82080829,-1.87265269,3.34199444,0.09530707,-3.77394007,1.68183889,1.97630386,1.48839111,-3.00986825,4.13786954,1.46723819,8.08842927,10.94846901,2.22280215,-1.85681443,4.47327707,3.15918201,-0.77879694,-0.11557772,0.21580221,-2.62405631,-3.42924226,-7.01213438,7.75544419,-3.72408571,3.46613566};
//int test_result_expected_label [NUMBER_OF_TEST_VECTORS*NUMBER_OF_OUTPUT_WORDS] = {0x03,0x02, 0x01, 0x01}; // 1 outputs *2
int result_memory [NUMBER_OF_TEST_VECTORS*NUMBER_OF_OUTPUT_WORDS]; // same size as test_result_expected_memory
float results [NUMBER_OF_TEST_VECTORS][NUMBER_OF_OUTPUT_WORDS];
float mlp_labels [NUMBER_OF_TEST_VECTORS];
/*****************************************************************************
* Main function
******************************************************************************/
int main()
{
	int word_cnt, test_case_cnt = 0;
	int success;
	AXIS_wLAST read_output, write_input;
	hls::stream<AXIS_wLAST> S_AXIS;
	hls::stream<AXIS_wLAST> M_AXIS;

	/************** Run a software version of the hardware function to validate results ************/
	// instead of hard-coding the results in test_result_expected_memory

	for (test_case_cnt=0 ; test_case_cnt < NUMBER_OF_TEST_VECTORS ; test_case_cnt++){


		/******************** Input to Coprocessor : Transmit the Data Stream ***********************/

		printf(" Transmitting Data for test case %d ... \r\n", test_case_cnt);

		for (word_cnt=0 ; word_cnt < NUMBER_OF_INPUT_WORDS ; word_cnt++){

			write_input.data = test_input_memory[word_cnt];
//			write_input.data = test_input_memory[word_cnt+test_case_cnt*NUMBER_OF_INPUT_WORDS];
//			write_input.data = converter.i;
			write_input.last = 0;
			if(word_cnt==NUMBER_OF_INPUT_WORDS-1)
			{
				write_input.last = 1;
				// S_AXIS_TLAST is asserted for the last word.
				// Actually, doesn't matter since we are not making using of S_AXIS_TLAST.
			}
			S_AXIS.write(write_input); // insert one word into the stream
		}

		/* Transmission Complete */

		/********************* Call the hardware function (invoke the co-processor / ip) ***************/

		pca_mlp(S_AXIS, M_AXIS);


		/******************** Output from Coprocessor : Receive the Data Stream ***********************/

		printf(" Receiving data for test case %d ... \r\n", test_case_cnt);

		for (word_cnt=0 ; word_cnt < NUMBER_OF_OUTPUT_WORDS ; word_cnt++){

			read_output = M_AXIS.read(); // extract one word from the stream
//			converter.i = read_output.data;
			results[test_case_cnt][word_cnt] = read_output.data;
			printf("read_output: %f ...\n", read_output.data);
		}


		/* Reception Complete */
	}

	/************************** Checking correctness of results *****************************/

	success = 1;

	/* Compare the data send with the data received */


	/* Compare the data send with the data received */
//	printf(" Comparing data ...\r\n");
//	for(word_cnt=0; word_cnt < NUMBER_OF_TEST_VECTORS*NUMBER_OF_OUTPUT_WORDS; word_cnt++){
//		// success = success & (result_memory[word_cnt] == test_result_expected_memory[word_cnt]);
//		printf(" output data %d ; expected data %d ... \r\n", result_memory[word_cnt], test_result_expected_memory[word_cnt]);
//	}

	// if (success != 1){
	// 	printf("Test Failed\r\n");
	// 	return 1;
	// }

	// printf("Test Success\r\n");

	return 0;
}
