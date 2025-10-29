import pandas

def save_to_csv(data: list[tuple[int, int, int]], filename: str) -> None:
    """
    將資料存成 CSV 檔案。
    data: list of (ell, min_pop, lo, hi, nfe)
    """
    df = pandas.DataFrame(data, columns=['ell', 'min_pop', 'avg_nfe'])
    df.to_csv(filename, index=False)
    
if __name__ == "__main__":
    # read data from out.log
    with open('out_5_250.log', 'r') as f:
        lines = f.readlines()
    sample_data = []
    for line in lines:
        
        parts = line.strip().split(" ")
        if len(parts) == 8:
            ell = int(parts[1][4:-1])
            min_pop = int(parts[3][4:])
            avg_nfe = int(parts[7][4:])
            print(f"Parsed: ell={ell}, min_pop={min_pop}, avg_nfe={avg_nfe}")
            sample_data.append((ell, min_pop, avg_nfe))
    
    
    save_to_csv(sample_data, 'plot_result.csv')