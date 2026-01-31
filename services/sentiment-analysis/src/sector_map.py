"""
EGX30 Sector Map
Mapping sectors to their constituent tickers for sentiment analysis.
"""

SECTOR_MAP = {
    "Banks": ['COMI', 'ADIB', 'CIEB', 'QNBA', 'FAIT'],
    "Real Estate": ['TMGH', 'EMFD', 'PHDC', 'MASR', 'ORHD'],
    "Basic Resources": ['EGAL', 'ABUK', 'MFPC', 'SIDPEC', 'ESRS'],
    "Non-Bank Financials": ['HRHO', 'BTFH', 'FWRY', 'EFIH', 'CIRA'],
    "Consumer": ['EAST', 'JUFO', 'ORWE', 'AUTO'],
    "Telecom": ['ETEL']
}

# Reverse map for easy lookup: Ticker -> Sector
TICKER_TO_SECTOR = {}
for sector, tickers in SECTOR_MAP.items():
    for ticker in tickers:
        TICKER_TO_SECTOR[ticker] = sector

# Aliases for text matching
TICKER_ALIASES = {
    'COMI': ['CIB', 'Commercial International Bank'],
    'ADIB': ['Abu Dhabi Islamic Bank'],
    'CIEB': ['Credit Agricole'],
    'QNBA': ['QNB Alahli'],
    'FAIT': ['Faisal Islamic Bank'],
    'TMGH': ['Talaat Moustafa', 'TMG'],
    'EMFD': ['Emaar Misr'],
    'PHDC': ['Palm Hills'],
    'MASR': ['Madinet Masr', 'MNHD'],
    'ORHD': ['Orascom Development'],
    'EGAL': ['Egypt Aluminum'],
    'ABUK': ['Abu Qir Fertilizers'],
    'MFPC': ['MOPCO'],
    'SIDPEC': ['Sidi Kerir'],
    'ESRS': ['Ezz Steel'],
    'HRHO': ['EFG Hermes'],
    "BTFH.CA": ["Beltone", "BTFH", "Beltone Financial", "بلتون", "بلتون المالية"],
    'EFIH': ['e-finance'],
    'CIRA': ['Cairo for Investment'],
    'EAST': ['Eastern Company'],
    'JUFO': ['Juhayna'],
    'ORWE': ['Oriental Weavers'],
    'AUTO': ['GB Corp', 'Ghabbour'],
    'ETEL': ['Telecom Egypt', 'WE']
}
