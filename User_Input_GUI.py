import tkinter as tk
from tkinter import messagebox, ttk
import pandas as pd

cancelled = False  # Flag for cancellation

def launch_multi_entry_gui():
    collected_data = []

    def add_entry():
        try:
            entry = {
                "Client Name": name_entry.get(),
                "Client e-mail": email_entry.get(),
                "Profession": profession_entry.get(),
                "Education": education_entry.get(),
                "Country": country_entry.get(),
                "Gender": gender_var.get(),
                "Age": float(age_entry.get()),
                "Income": float(income_entry.get()),
                "Credit Card Debt": float(debt_entry.get()),
                "Healthcare Cost": float(healthcare_entry.get()),
                "Inherited Amount": float(inherited_entry.get()),
                "Stocks": float(stocks_entry.get()),
                "Bonds": float(bonds_entry.get()),
                "Mutual Funds": float(mutual_funds_entry.get()),
                "ETFs": float(etfs_entry.get()),
                "REITs": float(reits_entry.get()),
                "Net Worth": float(worth_entry.get())
            }
            collected_data.append(entry)
            update_status()
            clear_form()
        except ValueError as e:
            messagebox.showerror("Invalid input", f"Please check your entries.\n{e}")

    def update_status():
        status_var.set(f"Entries added: {len(collected_data)}")

    def clear_form():
        name_entry.delete(0, tk.END)
        email_entry.delete(0, tk.END)
        profession_entry.delete(0, tk.END)
        education_entry.delete(0, tk.END)
        country_entry.delete(0, tk.END)
        gender_var.set("Male")
        age_entry.delete(0, tk.END)
        income_entry.delete(0, tk.END)
        debt_entry.delete(0, tk.END)
        healthcare_entry.delete(0, tk.END)
        inherited_entry.delete(0, tk.END)
        stocks_entry.delete(0, tk.END)
        bonds_entry.delete(0, tk.END)
        mutual_funds_entry.delete(0, tk.END)
        etfs_entry.delete(0, tk.END)
        reits_entry.delete(0, tk.END)
        worth_entry.delete(0, tk.END)

    def finish():
        if collected_data:
            root.destroy()
        else:
            if messagebox.askyesno("No entries", "No entries added. Do you want to exit?"):
                collected_data.clear()
                root.destroy()

    def cancel():
        if collected_data:
            if messagebox.askyesno("Confirm", "You have unsaved entries. Do you want to cancel and keep them?"):
                root.destroy()
            else:
                return  # Do nothing, stay in GUI
        else:
            root.destroy()

    root = tk.Tk()
    root.title("Net Worth Prediction - Data Entry")
    root.geometry("500x600")
    root.configure(bg='#f0f0f0')

    # Main frame
    main_frame = ttk.Frame(root, padding="10")
    main_frame.pack(fill="both", expand=True)

    # Title
    title_label = tk.Label(main_frame, text="Net Worth Prediction", 
                          font=("Arial", 16, "bold"), bg='#f0f0f0')
    title_label.pack(pady=(0, 20))

    # Create scrollable frame
    canvas = tk.Canvas(main_frame, bg='#f0f0f0')
    scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    # Personal Information Section
    personal_frame = ttk.LabelFrame(scrollable_frame, text="Personal Information", padding="10")
    personal_frame.pack(fill="x", pady=(0, 10))

    # Personal fields
    name_entry = tk.Entry(personal_frame, width=25)
    email_entry = tk.Entry(personal_frame, width=25)
    profession_entry = tk.Entry(personal_frame, width=25)
    education_entry = tk.Entry(personal_frame, width=25)
    country_entry = tk.Entry(personal_frame, width=25)
    gender_var = tk.StringVar(value="Male")
    age_entry = tk.Entry(personal_frame, width=25)

    # Grid layout for personal info
    tk.Label(personal_frame, text="Name:").grid(row=0, column=0, sticky="e", padx=5, pady=3)
    name_entry.grid(row=0, column=1, padx=5, pady=3)
    
    tk.Label(personal_frame, text="Email:").grid(row=1, column=0, sticky="e", padx=5, pady=3)
    email_entry.grid(row=1, column=1, padx=5, pady=3)
    
    tk.Label(personal_frame, text="Profession:").grid(row=2, column=0, sticky="e", padx=5, pady=3)
    profession_entry.grid(row=2, column=1, padx=5, pady=3)
    
    tk.Label(personal_frame, text="Education:").grid(row=3, column=0, sticky="e", padx=5, pady=3)
    education_entry.grid(row=3, column=1, padx=5, pady=3)
    
    tk.Label(personal_frame, text="Country:").grid(row=4, column=0, sticky="e", padx=5, pady=3)
    country_entry.grid(row=4, column=1, padx=5, pady=3)
    
    tk.Label(personal_frame, text="Gender:").grid(row=5, column=0, sticky="e", padx=5, pady=3)
    tk.OptionMenu(personal_frame, gender_var, "Male", "Female").grid(row=5, column=1, padx=5, pady=3, sticky="w")
    
    tk.Label(personal_frame, text="Age:").grid(row=6, column=0, sticky="e", padx=5, pady=3)
    age_entry.grid(row=6, column=1, padx=5, pady=3)

    # Financial Information Section
    financial_frame = ttk.LabelFrame(scrollable_frame, text="Financial Information", padding="10")
    financial_frame.pack(fill="x", pady=(0, 10))
    
    income_entry = tk.Entry(financial_frame, width=25)
    debt_entry = tk.Entry(financial_frame, width=25)
    healthcare_entry = tk.Entry(financial_frame, width=25)
    inherited_entry = tk.Entry(financial_frame, width=25)
    worth_entry = tk.Entry(financial_frame, width=25)

    # Grid layout for financial info
    tk.Label(financial_frame, text="Income:").grid(row=0, column=0, sticky="e", padx=5, pady=3)
    income_entry.grid(row=0, column=1, padx=5, pady=3)
    
    tk.Label(financial_frame, text="Credit Card Debt:").grid(row=1, column=0, sticky="e", padx=5, pady=3)
    debt_entry.grid(row=1, column=1, padx=5, pady=3)
    
    tk.Label(financial_frame, text="Healthcare Cost:").grid(row=2, column=0, sticky="e", padx=5, pady=3)
    healthcare_entry.grid(row=2, column=1, padx=5, pady=3)
    
    tk.Label(financial_frame, text="Inherited Amount:").grid(row=3, column=0, sticky="e", padx=5, pady=3)
    inherited_entry.grid(row=3, column=1, padx=5, pady=3)
    
    tk.Label(financial_frame, text="Net Worth:").grid(row=4, column=0, sticky="e", padx=5, pady=3)
    worth_entry.grid(row=4, column=1, padx=5, pady=3)

    # Investment Information Section
    investment_frame = ttk.LabelFrame(scrollable_frame, text="Investment Portfolio", padding="10")
    investment_frame.pack(fill="x", pady=(0, 10))
    
    stocks_entry = tk.Entry(investment_frame, width=25)
    bonds_entry = tk.Entry(investment_frame, width=25)
    mutual_funds_entry = tk.Entry(investment_frame, width=25)
    etfs_entry = tk.Entry(investment_frame, width=25)
    reits_entry = tk.Entry(investment_frame, width=25)

    # Grid layout for investment info
    tk.Label(investment_frame, text="Stocks:").grid(row=0, column=0, sticky="e", padx=5, pady=3)
    stocks_entry.grid(row=0, column=1, padx=5, pady=3)
    
    tk.Label(investment_frame, text="Bonds:").grid(row=1, column=0, sticky="e", padx=5, pady=3)
    bonds_entry.grid(row=1, column=1, padx=5, pady=3)
    
    tk.Label(investment_frame, text="Mutual Funds:").grid(row=2, column=0, sticky="e", padx=5, pady=3)
    mutual_funds_entry.grid(row=2, column=1, padx=5, pady=3)
    
    tk.Label(investment_frame, text="ETFs:").grid(row=3, column=0, sticky="e", padx=5, pady=3)
    etfs_entry.grid(row=3, column=1, padx=5, pady=3)
    
    tk.Label(investment_frame, text="REITs:").grid(row=4, column=0, sticky="e", padx=5, pady=3)
    reits_entry.grid(row=4, column=1, padx=5, pady=3)

    # Status and buttons
    status_var = tk.StringVar()
    status_label = tk.Label(scrollable_frame, textvariable=status_var, bg='#f0f0f0', font=("Arial", 10, "bold"))
    status_label.pack(pady=10)
    update_status()

    button_frame = tk.Frame(scrollable_frame, bg='#f0f0f0')
    button_frame.pack(pady=10)
    
    tk.Button(button_frame, text="Add Entry", command=add_entry, 
              bg="lightgreen", width=12, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Finish", command=finish, 
              bg="lightblue", width=12, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
    tk.Button(button_frame, text="Cancel", command=cancel, 
              bg="lightcoral", width=12, font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

    # Pack the canvas and scrollbar
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    root.mainloop()

    if collected_data:
        return pd.DataFrame(collected_data)
    else:
        return None


# Usage example:
if __name__ == "__main__":
    df = launch_multi_entry_gui()
    if df is not None:
        print("All user inputs:")
        print(df)
    else:
        print("No data was entered.")
