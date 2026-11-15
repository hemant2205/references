
# Fractional Knapsack using Greedy Method
#keep wt less than profit in input


# Fractional Knapsack using Greedy Method

n = int(input("Enter number of items: "))

items = []

for i in range(n):
    weight = int(input("Enter weight: "))
    profit = int(input("Enter profit: "))

    ratio = profit / weight
    items.append([i + 1, weight, profit, ratio])

capacity = int(input("Enter knapsack capacity: "))

# Display all items
print("\nAll Items:")
for item in items:
    print("Item:", item[0], "Weight:", item[1],
          "Profit:", item[2], "Ratio:", round(item[3], 2))

# Sort items by profit/weight ratio
items.sort(key=lambda x: x[3], reverse=True)

total_profit = 0

print("\nSelected Items:")

# Select items
for item in items:

    number = item[0]
    weight = item[1]
    profit = item[2]
    ratio = item[3]

    if capacity >= weight:
        total_profit += profit
        capacity -= weight
        print("Item:", number, "Weight Taken:", weight,
              "Profit:", profit)

    else:
        taken = capacity
        earned = ratio * taken
        total_profit += earned

        print("Item:", number, "Weight Taken:", taken,
              "out of", weight, "Profit:", round(earned, 2))
        break

print("\nMaximum Profit:", round(total_profit, 2))

