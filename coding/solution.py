# filename: solution.py
def solution(d, budget):
    d.sort()  # Step 1: Sort the requests
    count = 0  # Step 2: Initialize the count of supported departments
    total_spent = 0  # Initialize total spent amount
    
    for request in d:  # Step 3: Iterate through sorted requests
        if total_spent + request <= budget:  # Check if we can support this request
            total_spent += request  # Update total spent
            count += 1  # Increment the count of supported departments
        else:
            break  # If we can't support this request, break the loop
    
    return count  # Step 4: Return the count of supported departments

# Example usage
print(solution([1, 3, 2, 5, 4], 9))  # Output: 3
print(solution([2, 2, 3, 3], 10))    # Output: 4