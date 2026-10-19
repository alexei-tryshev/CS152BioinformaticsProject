#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINES 1000
#define MAX_LENGTH 256

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Error: please enter the command properly.\n");
        return 1;
    }

    char *command = argv[1];    //get the command
    char *fileName = argv[2];   //get the file
    int n = atoi(argv[3]);     //convert char to int

    FILE* fptr;
    fptr = fopen(fileName, "r");

    if (fptr == NULL) {
        printf("The file doesn't exist");
    }

    char content[MAX_LINES][MAX_LENGTH];
    // this = getting the content of the file

    int totalLines = 0;

    while (fgets(content[totalLines], MAX_LENGTH, fptr) != NULL) {
        totalLines++;

        if (totalLines >= MAX_LINES) {
            break;
        }
    }
    fclose(fptr);

    if (strcmp(command, "tail") == 0) {

        // if user calls for the whole file (n > length of the file)
        if (n > totalLines) {
            n = totalLines;
        }

        int start = totalLines - n;
        // this = getting the start of the file

        for (int i = start; i < totalLines; i++) {
            printf("%s", content[i]);
        }
    }

    else if (strcmp(command, "first") == 0) {
        if (n > totalLines) {
            n = totalLines;
        }
        
        for (int i = 0; i < n; i++) {
            printf("%s", content[i]);
        }
    }

    else if (strcmp(command, "middle") == 0) {
        int start = (totalLines - n) / 2;

        for (int i = start; i < start + n; i++) {
            printf("%s", content[i]);
        }
    }

    

    return 0;
}