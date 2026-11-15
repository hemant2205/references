
# 0/1 Knapsack using Dynamic Programming

n = int(input("Enter number of items: "))

weight = []
profit = []

for i in range(n):
    weight.append(int(input("Enter weight: ")))
    profit.append(int(input("Enter profit: ")))

capacity = int(input("Enter knapsack capacity: "))

# Display all items
print("\nAll Items:")
for i in range(n):
    print("Item:", i + 1, "Weight:", weight[i], "Profit:", profit[i])

# Create DP table
dp = [[0 for j in range(capacity + 1)] for i in range(n + 1)]

# Fill DP table
for i in range(1, n + 1):
    for w in range(1, capacity + 1):

        if weight[i - 1] <= w:
            dp[i][w] = max(
                profit[i - 1] + dp[i - 1][w - weight[i - 1]],
                dp[i - 1][w]
            )
        else:
            dp[i][w] = dp[i - 1][w]

# Display selected items
print("\nSelected Items:")

w = capacity
for i in range(n, 0, -1):
    if dp[i][w] != dp[i - 1][w]:
        print("Item:", i, "Weight:", weight[i - 1],
              "Profit:", profit[i - 1])
        w -= weight[i - 1]

# Display maximum profit
print("\nMaximum Profit:", dp[n][capacity])
