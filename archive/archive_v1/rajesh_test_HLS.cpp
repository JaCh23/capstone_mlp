
#include <stdio.h>
#include <math.h>
#include "hls_stream.h"

/***************** AXIS with TLAST structure declaration *********************/

struct AXIS_wLAST{
	int data;
	bool last;
};

/***************** Coprocessor function declaration *********************/

void myip_v1_0_HLS(hls::stream<AXIS_wLAST>& S_AXIS, hls::stream<AXIS_wLAST>& M_AXIS);


/***************** Macros *********************/
#define NUMBER_OF_INPUT_WORDS 4  // length of an input vector
#define NUMBER_OF_OUTPUT_WORDS 4  // length of an input vector
#define NUMBER_OF_TEST_VECTORS 2  // number of such test vectors (cases)


/************************** Variable Definitions *****************************/
int test_input_memory [NUMBER_OF_TEST_VECTORS*NUMBER_OF_INPUT_WORDS] = {0x01, 0x02, 0x03, 0x04, 0x02, 0x03, 0x04, 0x05}; // 4 inputs * 2
int test_result_expected_memory [NUMBER_OF_TEST_VECTORS*NUMBER_OF_OUTPUT_WORDS];// 4 outputs *2
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
	int sum;
	for (test_case_cnt=0 ; test_case_cnt < NUMBER_OF_TEST_VECTORS ; test_case_cnt++){
		sum = 0;
		for (word_cnt=0 ; word_cnt < NUMBER_OF_INPUT_WORDS ; word_cnt++){
			sum +=test_input_memory[word_cnt+test_case_cnt*NUMBER_OF_INPUT_WORDS];
		}
		for (word_cnt=0; word_cnt < NUMBER_OF_OUTPUT_WORDS; word_cnt++) {
			test_result_expected_memory[word_cnt+test_case_cnt*NUMBER_OF_OUTPUT_WORDS] = sum;
		}
	}


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

		myip_v1_0_HLS(S_AXIS, M_AXIS);


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
