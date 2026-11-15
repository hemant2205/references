
# Quick Sort - Deterministic and Randomized

import random

# Deterministic Quick Sort
def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[0]

    left = [x for x in arr[1:] if x <= pivot]
    right = [x for x in arr[1:] if x > pivot]

    return quick_sort(left) + [pivot] + quick_sort(right)


# Randomized Quick Sort
def randomized_quick_sort(arr):
    if len(arr) <= 1:
        return arr

    # Select random pivot
    pivot = random.choice(arr)

    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x > pivot]
    equal = [x for x in arr if x == pivot]

    return (randomized_quick_sort(left) +
            equal +
            randomized_quick_sort(right))


# Main Program
n = int(input("Enter number of elements: "))

arr = []
for i in range(n):
    arr.append(int(input("Enter element: ")))

print("Original Array:", arr)

print("Deterministic Quick Sort:", quick_sort(arr))

print("Randomized Quick Sort:", randomized_quick_sort(arr))
