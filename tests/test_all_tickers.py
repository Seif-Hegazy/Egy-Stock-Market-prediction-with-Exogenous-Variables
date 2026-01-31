import sys
import os

# --- Runtime Dependency Hotfix ---
LIB_DIR = "/app/src/libs"
if LIB_DIR not in sys.path:
    sys.path.insert(0, LIB_DIR)

import yfinance as yf
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Copy of STOCK_DATA from app.py
STOCK_DATA = {
    "Commercial International Bank (CIB)": {"ticker": "COMI.CA"},
    "Crédit Agricole Egypt": {"ticker": "CIEB.CA"},
    "Housing & Development Bank": {"ticker": "HDBK.CA"},
    "Faisal Islamic Bank of Egypt": {"ticker": "FAIT.CA"},
    "Abu Dhabi Islamic Bank (ADIB)": {"ticker": "ADIB.CA"},
    "Al Baraka Bank Egypt": {"ticker": "SAUD.CA"},
    "Egyptian Gulf Bank (EGBANK)": {"ticker": "EGBE.CA"},
    "Export Development Bank of Egypt (EBank)": {"ticker": "EXPA.CA"},
    "EFG Hermes": {"ticker": "HRHO.CA"},
    "E-Finance": {"ticker": "EFIH.CA"},
    "Fawry": {"ticker": "FWRY.CA"},
    "Beltone Financial": {"ticker": "BTFH.CA"},
    "CI Capital": {"ticker": "CICH.CA"},
    "Talaat Moustafa Group (TMG)": {"ticker": "TMGH.CA"},
    "Palm Hills Developments": {"ticker": "PHDC.CA"},
    "Sixth of October Development & Investment (SODIC)": {"ticker": "OCDI.CA"},
    "Madinet Masr (MNHD)": {"ticker": "MASR.CA"},
    "Heliopolis Housing": {"ticker": "HELI.CA"},
    "Orascom Construction": {"ticker": "ORAS.CA"},
    "Emaar Misr": {"ticker": "EMFD.CA"},
    "Elsewedy Electric": {"ticker": "SWDY.CA"},
    "Abu Qir Fertilizers": {"ticker": "ABUK.CA"},
    "Misr Fertilizers Production (MOPCO)": {"ticker": "MFPC.CA"},
    "Sidi Kerir Petrochemicals (SIDPEC)": {"ticker": "SKPC.CA"},
    "Alexandria Mineral Oils (AMOC)": {"ticker": "AMOC.CA"},
    "Telecom Egypt (WE)": {"ticker": "ETEL.CA"},
    "Eastern Company": {"ticker": "EAST.CA"},
    "Juhayna Food Industries": {"ticker": "JUFO.CA"},
    "Edita Food Industries": {"ticker": "EFID.CA"},
    "Ibnsina Pharma": {"ticker": "ISPH.CA"},
    "Cleopatra Hospitals": {"ticker": "CLHO.CA"},
    "GB Corp (Ghabbour)": {"ticker": "GBCO.CA"},
    "Egypt Kuwait Holding": {"ticker": "EKHO.CA"},
    "Qalaa Holdings": {"ticker": "CCAP.CA"},
    "Egyptian Satellites (NileSat)": {"ticker": "EGSA.CA"}
}

print(f"Testing {len(STOCK_DATA)} tickers...")
print("-" * 60)
print(f"{'Ticker':<10} | {'Status':<10} | {'Rows':<5} | {'Last Date'}")
print("-" * 60)

success_count = 0
fail_count = 0

for name, data in STOCK_DATA.items():
    ticker_symbol = data["ticker"]
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1mo")
        
        if hist.empty:
            print(f"{ticker_symbol:<10} | ❌ FAIL     | 0     | N/A")
            fail_count += 1
        else:
            last_date = hist.index[-1].strftime('%Y-%m-%d')
            print(f"{ticker_symbol:<10} | ✅ OK       | {len(hist):<5} | {last_date}")
            success_count += 1
            
    except Exception as e:
        print(f"{ticker_symbol:<10} | ⚠️ ERROR    | 0     | {str(e)}")
        fail_count += 1

print("-" * 60)
print(f"Summary: {success_count} Success, {fail_count} Failed")
