
#include <stdio.h>
#include <math.h>
#include "hls_stream.h"

/***************** AXIS with TLAST structure declaration *********************/

struct AXIS_wLAST{
	int data;
	bool last;
};


/***************** Coprocessor function declaration *********************/

void forward_pass_mlp(hls::stream<AXIS_wLAST>& S_AXIS, hls::stream<AXIS_wLAST>& M_AXIS);


/***************** Macros *********************/
#define NUMBER_OF_INPUT_WORDS 24  // length of an input vector
#define NUMBER_OF_OUTPUT_WORDS 1  // length of an input vector
#define NUMBER_OF_TEST_VECTORS 5  // number of such test vectors (cases)

#define NUM_INPUTS 2
#define NUM_HIDDEN 2
#define NUM_OUTPUTS 2

/************************** Variable Definitions *****************************/
int test_input_memory [NUMBER_OF_TEST_VECTORS*NUMBER_OF_INPUT_WORDS] = {0.017314464190467227, 0.7821381516657591, 1.0, 0.7851502881374393, 0.8214685624410034, 0.777309756018898, 1.0, 0.7438152161080576, 0.8758875965614796, 0.0, 0.8176854178020512, 0.6521847636448815, 0.9915502727104807, 0.14720052571656392, 0.2867605526436922, 0.0, 0.0, 1.0, 0.0, 0.8176854178020512, 0.6521847636448815, 1.0, 0.0, 0.0, 0.7327470560085914, 0.7670054540398284, 0.7876194904561352, 0.0003777085979821404, 0.004562143134149377, 0.015229133588794186, 0.008489068444362813, 0.0, 0.009446904812559165, 0.9824486506311336, 0.5290985126034317, 0.38614894284325796, 0.9494529308687075, 0.0, 0.07397361806125458, 0.59476694098463, 0.7198436209281258, 0.26609692233259596, 0.9824486506311336, 0.5290985126034317, 0.38614894284325796, 0.5751971490522738, 0.5788592538242026, 0.6546772353702576, 0.6303608153040674, 0.6459125042423195, 0.7726308539512028, 0.009468520442085682, 0.011712710762579626, 0.017824833108515348, 0.005093946644663561, 0.019465654191712645, 0.015376622627928382, 0.83442008939342, 0.41204243574476007, 0.9999999999999999, 0.9916250943994421, 0.3075444576465419, 0.3291439600701033, 0.5410450286608446, 0.665238985485304, 0.1066198430894188, 0.83442008939342, 0.41204243574476007, 0.9999999999999999, 0.5787903934852892, 0.6250082118276563, 0.5450281584503717, 0.6522639521708093, 0.5505606440131224, 0.7592430676622473, 0.0, 0.0, 0.0, 0.0, 0.0047103240218238085, 0.0, 0.7469269261405295, 0.6540769780819387, 0.48900901327472834, 1.0, 0.2661700890020115, 0.2057585278383944, 0.583041640151744, 0.7457313758113601, 0.09223203419799791, 0.7469269261405295, 0.6540769780819387, 0.48900901327472834, 0.577763837422617, 0.5693925573430506, 0.5870560088071526, 0.0, 0.7878431164406379, 0.6870106909630802, 0.1287997530906948, 0.04890807975145384, 0.11378937234541442, 0.07753320210333703, 0.3867255070794684, 0.12348765216436264, 0.8663248888827084, 1.0, 0.28347135356732167, 0.2654557598637883, 0.9525514116146542, 0.4973485965097738, 0.36541609268994457, 0.8711934524925733, 0.11531701824106748, 0.8663248888827084, 1.0, 0.28347135356732167, 0.5362222766191846, 0.4115857831263833, 0.8471261015945116}; // 4 inputs * 2
int test_result_expected_memory [NUMBER_OF_TEST_VECTORS*NUMBER_OF_OUTPUT_WORDS] = {0x03, 0x02, 0x01, 0x01, 0x00}; // 1 outputs *2
int result_memory [NUMBER_OF_TEST_VECTORS*NUMBER_OF_OUTPUT_WORDS]; // same size as test_result_expected_memory

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

			write_input.data = test_input_memory[word_cnt+test_case_cnt*NUMBER_OF_INPUT_WORDS];
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

		forward_pass_mlp(S_AXIS, M_AXIS);


		/******************** Output from Coprocessor : Receive the Data Stream ***********************/

		printf(" Receiving data for test case %d ... \r\n", test_case_cnt);

		for (word_cnt=0 ; word_cnt < NUMBER_OF_OUTPUT_WORDS ; word_cnt++){

			read_output = M_AXIS.read(); // extract one word from the stream
			result_memory[word_cnt+test_case_cnt*NUMBER_OF_OUTPUT_WORDS] = read_output.data;
		}

		/* Reception Complete */
	}

	/************************** Checking correctness of results *****************************/

	success = 1;

	/* Compare the data send with the data received */
	printf(" Comparing data ...\r\n");
	for(word_cnt=0; word_cnt < NUMBER_OF_TEST_VECTORS*NUMBER_OF_OUTPUT_WORDS; word_cnt++){
		success = success & (result_memory[word_cnt] == test_result_expected_memory[word_cnt]);
	}

	if (success != 1){
		printf("Test Failed\r\n");
		return 1;
	}

	printf("Test Success\r\n");

	return 0;
}
