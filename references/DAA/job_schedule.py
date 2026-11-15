
# Job Sequencing using Greedy Method

n = int(input("Enter number of jobs: "))

jobs = []

for i in range(n):
    job = input("Enter job name: ")
    deadline = int(input("Enter deadline: "))
    profit = int(input("Enter profit: "))

    jobs.append([job, deadline, profit])

# Sort jobs by profit in descending order
jobs.sort(key=lambda x: x[2], reverse=True)

# Find maximum deadline
max_deadline = max(job[1] for job in jobs)

# Create empty time slots
slots = [None] * max_deadline

total_profit = 0

# Schedule jobs
for job in jobs:
    for j in range(job[1] - 1, -1, -1):

        if slots[j] is None:
            slots[j] = job[0]
            total_profit += job[2]
            break

# Display result
print("\nScheduled Jobs:")
for job in slots:
    if job is not None:
        print(job, end=" ")

print("\nTotal Profit:", total_profit)
