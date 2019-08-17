#include "ConsoleLog.h"
#include <assert.h>
#include <Windows.h>
#include <stdio.h>
#include <stdarg.h>

void ConsoleLog(const char * level, const char * i_fmt, ...)
{
	assert(i_fmt);
	assert(level);

	const size_t lenTemp = 256;
	char strTemp[lenTemp] = "";

	strcat_s(strTemp, level);
	strcat_s(strTemp, ": ");

	strcat_s(strTemp, i_fmt);
	strcat_s(strTemp, "\n");

	const size_t lenOutput = lenTemp + 32;

	char strOutput[lenOutput];

	// define a variable argument list variable 
	va_list args;

	// initialize it. second parameter is variable immediately
	// preceeding variable arguments
	va_start(args, i_fmt);

	// (safely) print our formatted string to a char buffer
	vsprintf_s(strOutput, lenOutput, strTemp, args);

	va_end(args);

	OutputDebugStringA(strOutput);
}

