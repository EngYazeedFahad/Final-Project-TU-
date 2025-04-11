#Eng\Yazeed fahad alzaidi
#Eng\Mohammed Alhumaidi
#Eng\Nawaf bin shafi
#Eng\Abdulmohsen Alghamdi
#Eng\Waleed abdi
#Eng\Mohammed Altwirqi
#Project Advisor Prof. Httan Ali Asiri
#---------------------------------------------------------------------------------------------------------------------#
#Graphical user interface (GUI):
def main():
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk
    import numpy as np
    import joblib

    # Load model and scaler
    model = joblib.load("voting.sav")
    scaler = joblib.load("scaler_for_voting.sav")

    # Create main window
    root = tk.Tk()
    root.title("Diabetes Complication Predictor")
    root.geometry("750x720")
    root.configure(bg="#f0faff")  # Updated soft white-blue

    # Feature names
    feature_names = [
        "GenderID", "Age", "Temp",
        "BP DIASTOLIC", "BP SYSTOLIC",
        "PULSE", "Height", "Weight"
    ]

    # Gender map
    gender_map = {"Male": 0, "Female": 1}

    # Styling (modern look)
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TLabel", font=("Segoe UI", 11), background="#f0faff")
    style.configure("TEntry", font=("Segoe UI", 11))
    style.configure("TButton", font=("Segoe UI", 11, "bold"), background="#4a90e2", foreground="white")
    style.configure("TCombobox", font=("Segoe UI", 11))
    style.configure("TLabelframe", background="#ffffff", borderwidth=0)
    style.configure("TLabelframe.Label", font=("Segoe UI", 12, "bold"), background="#ffffff")
    style.configure("TFrame", background="#ffffff")

    # ========== Header with Logos ==========
    header_border = tk.Frame(root, bg="#f0faff", highlightbackground="black", highlightthickness=2)
    header_border.pack(padx=40, pady=(20, 10), fill="x")

    header_frame = tk.Frame(header_border, bg="#f0faff")
    header_frame.pack(fill="x", pady=10)

    left_img = Image.open("helth.png").resize((90, 100), Image.Resampling.LANCZOS)
    left_photo = ImageTk.PhotoImage(left_img)
    tk.Label(header_frame, image=left_photo, bg="#f0faff").pack(side="left", padx=20)

    tk.Label(
        header_frame,
        text="Patient Diabetes Risk Checker",
        font=("Segoe UI", 20, "bold"),
        bg="#f0faff",
        fg="#333333"
    ).pack(side="left", expand=True)

    right_img = Image.open("TU.png").resize((90, 100), Image.Resampling.LANCZOS)
    right_photo = ImageTk.PhotoImage(right_img)
    tk.Label(header_frame, image=right_photo, bg="#f0faff").pack(side="right", padx=20)

    # ========== Form Section ==========
    form_outer = tk.Frame(root, bg="#f0faff", highlightbackground="black", highlightthickness=2)
    form_outer.pack(padx=40, pady=15, fill='x')

    form_border = ttk.LabelFrame(form_outer, text="Enter Patient Data", padding="20 15 20 15")
    form_border.pack(fill='x')

    form_inner = ttk.Frame(form_border)
    form_inner.pack(fill='both', expand=True)
    form_inner.grid_columnconfigure(0, weight=1)
    form_inner.grid_columnconfigure(1, weight=1)

    entries = []
    gender_display = tk.StringVar(value="Select")

    for idx, name in enumerate(feature_names):
        label = ttk.Label(form_inner, text=name + ":", anchor='center')
        label.grid(row=idx, column=0, padx=10, pady=8, sticky="e")

        if name == "GenderID":
            gender_combo = ttk.Combobox(form_inner, textvariable=gender_display, state="normal", width=27)
            gender_combo['values'] = ["Select", "Male", "Female"]
            gender_combo.grid(row=idx, column=1, padx=10, pady=8, sticky="w")
            entries.append(gender_display)
        else:
            entry = ttk.Entry(form_inner, width=30)
            entry.grid(row=idx, column=1, padx=10, pady=8, sticky="w")
            entries.append(entry)

    # ========== Prediction Result Section ==========
    result_outer = tk.Frame(root, bg="#f0faff", highlightbackground="black", highlightthickness=2)
    result_outer.pack(padx=40, pady=20, fill='x')

    result_frame = ttk.LabelFrame(result_outer, text="Prediction Result", padding="20 10 20 10")
    result_frame.pack(fill='x')

    result_label = ttk.Label(
        result_frame,
        text="Result will appear here.",
        font=("Segoe UI", 12),
        background="#ffffff",
        foreground="#444444"
    )
    result_label.pack(pady=5)

    # ========== Predict Button ==========
    def predict():
        try:
            input_data = []
            for e in entries:
                if isinstance(e, tk.StringVar):
                    g = e.get()
                    if g not in gender_map:
                        result_label.config(foreground="red", text="Please select Gender: Male or Female")
                        return
                    input_data.append(float(gender_map[g]))
                else:
                    input_data.append(float(e.get()))

            sample = np.array([input_data])
            sample_scaled = scaler.transform(sample)
            prediction = model.predict(sample_scaled)[0]

            if prediction == 1:
                result_label.config(foreground="red", text="Prediction: With Complication")
            else:
                result_label.config(foreground="green", text="Prediction: Without Complication")

        except Exception as ex:
            result_label.config(foreground="red", text=f"Error: {str(ex)}")

    # Stylish button with space
    button_frame = tk.Frame(root, bg="#f0faff")
    button_frame.pack(pady=10)
    predict_btn = ttk.Button(button_frame, text="Predict", command=predict)
    predict_btn.pack(ipadx=15, ipady=5)

    # Run the app
    root.mainloop()

main()