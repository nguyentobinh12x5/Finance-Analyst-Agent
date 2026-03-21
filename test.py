def calculate_investment_return(ticker, price, quantity):
    # Đặt Breakpoint ở dòng dưới đây (dòng 4)
    initial_value = price * quantity
    
    # Giả sử giá tăng 10%
    new_price = price * 1.1 
    current_value = new_price * quantity
    
    profit = current_value - initial_value
    
    print(f"Ticker: {ticker}")
    print(f"Profit: {profit}")
    
    return profit

# Gọi hàm để test
my_profit = calculate_investment_return("VRE", 25000, 1000)