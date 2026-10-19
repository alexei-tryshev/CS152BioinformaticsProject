package hw01;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class evenSquaresFilter {
    public static void main(String[] args) {
        int[] originalArray = {-10, -8, -6, -4, -2};
        int threshold = -9;

        int[] returnedThing = EvenSquaresFilter(originalArray, threshold);
        System.out.println("Array: " + Arrays.toString(returnedThing));
    }

    public static int[] EvenSquaresFilter(int[] array, int threshold) {
        ArrayList<Integer> evenNums = new ArrayList<>();
        ArrayList<Integer> sorted = new ArrayList<>();

        for (int i = 0; i < array.length; i++) {
            if (array[i] > threshold && array[i] % 2 == 0) {
                evenNums.add(array[i] * array[i]);
            }
        }
        Collections.sort(evenNums);

        for (int i = evenNums.size() - 1; i >= 0; i--) {
            if (i >= 1 && evenNums.get(i) == evenNums.get(i - 1)) {
                evenNums.remove(i);
            }
            else {
                sorted.add(evenNums.get(i));
            }
        }

        int[] sorted2 = sorted.stream().mapToInt(Integer::intValue).toArray();
        return sorted2;
    }
}
