# screener_data.py
# ~200 Indian NSE stocks organized by sector for stock screener

NIFTY_200_SECTORS = {
    "Banking & Finance": [
        "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "SBIN.NS", "BANKBARODA.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
        "INDUSINDBK.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS",
        "BANDHANBNK.NS", "RBLBANK.NS", "YESBANK.NS", "AUBANK.NS",
        "BAJFINANCE.NS", "BAJAJFINSV.NS", "MUTHOOTFIN.NS", "CHOLAFIN.NS",
        "M&MFIN.NS", "SHRIRAMFIN.NS", "LICHSGFIN.NS", "POONAWALLA.NS",
        "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS", "ICICIGI.NS",
        "NIACL.NS", "STARHEALTH.NS",
    ],
    "IT & Technology": [
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
        "LTIM.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS",
        "OFSS.NS", "KPITTECH.NS", "TATAELXSI.NS", "ROUTE.NS",
        "ZENSARTECH.NS", "HEXAWARE.NS", "NIIT.NS", "CYIENT.NS",
        "MASTEK.NS", "SONATSOFTW.NS", "INTELLECT.NS",
    ],
    "Consumer & FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
        "DABUR.NS", "MARICO.NS", "GODREJCP.NS", "COLPAL.NS",
        "EMAMILTD.NS", "TATACONSUM.NS", "VBL.NS", "UBL.NS",
        "MCDOWELL-N.NS", "RADICO.NS", "PGHH.NS", "GILLETTE.NS",
        "HONASA.NS", "ZOMATO.NS", "NYKAA.NS", "DEVYANI.NS",
        "JUBLFOOD.NS", "WESTLIFE.NS", "SAPPHIRE.NS",
    ],
    "Pharma & Healthcare": [
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
        "BIOCON.NS", "LUPIN.NS", "AUROPHARMA.NS", "TORNTPHARM.NS",
        "ALKEM.NS", "IPCALAB.NS", "ABBOTINDIA.NS", "PFIZER.NS",
        "GLAXO.NS", "SANOFI.NS", "NATCOPHARMA.NS", "GRANULES.NS",
        "LAURUSLABS.NS", "AJANTPHARM.NS", "JBCHEPHARM.NS",
        "APOLLOHOSP.NS", "FORTIS.NS", "MAXHEALTH.NS", "METROPOLIS.NS",
        "LALPATHLAB.NS", "THYROCARE.NS",
    ],
    "Auto & Ancillaries": [
        "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJAUTO.NS",
        "HEROMOTOCO.NS", "EICHERMOT.NS", "TVSMOTORS.NS", "ASHOKLEY.NS",
        "TVSMOTOR.NS", "ESCORTS.NS", "BOSCHLTD.NS", "MOTHERSON.NS",
        "BHARATFORG.NS", "APOLLOTYRE.NS", "CEATLTD.NS", "BALKRISIND.NS",
        "EXIDEIND.NS", "AMARAJABAT.NS", "SUNDRMFAST.NS", "MINDA.NS",
        "CRAFTSMAN.NS", "SUPRAJIT.NS",
    ],
    "Energy & Power": [
        "RELIANCE.NS", "ONGC.NS", "COALINDIA.NS", "NTPC.NS",
        "POWERGRID.NS", "BPCL.NS", "IOC.NS", "HINDPETRO.NS",
        "GAIL.NS", "IGL.NS", "MGL.NS", "GUJGASLTD.NS",
        "TATAPOWER.NS", "ADANIGREEN.NS", "ADANIPOWER.NS",
        "CESC.NS", "TORNTPOWER.NS", "JSPL.NS", "NHPC.NS",
        "SJVN.NS", "RECLTD.NS", "PFC.NS", "IREDA.NS",
    ],
    "Metals & Mining": [
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS",
        "SAIL.NS", "NMDC.NS", "MOIL.NS", "NATIONALUM.NS",
        "WELCORP.NS", "APL.NS", "RATNAMANI.NS", "JINDALSAW.NS",
    ],
    "Infrastructure & Real Estate": [
        "LT.NS", "ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS",
        "ACC.NS", "RAMCOCEM.NS", "JKCEMENT.NS", "HEIDELBERG.NS",
        "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS",
        "PHOENIXLTD.NS", "SOBHA.NS", "BRIGADE.NS", "SUNTECK.NS",
        "NAUKRI.NS", "IRCTC.NS", "CONCOR.NS", "GMRAIRPORT.NS",
    ],
    "Capital Goods & Industrials": [
        "SIEMENS.NS", "ABB.NS", "HAVELLS.NS", "VOLTAS.NS",
        "WHIRLPOOL.NS", "BLUESTARCO.NS", "CROMPTON.NS", "POLYCAB.NS",
        "KFINTECH.NS", "CAMS.NS", "CDSL.NS", "BSE.NS",
        "MCX.NS", "ANGELONE.NS", "5PAISA.NS", "MOTILALOFS.NS",
        "AAVAS.NS", "CANFINHOME.NS", "HOMEFIRST.NS",
    ],
    "Telecom & Media": [
        "BHARTIARTL.NS", "IDEA.NS", "TATACOMM.NS", "HFCL.NS",
        "ROUTE.NS", "ZEEL.NS", "SUNTV.NS", "NETWORK18.NS",
        "DISHTV.NS", "NAZARA.NS",
    ],
}

# Flat list for screener
ALL_SCREENER_TICKERS = [
    t for tickers in NIFTY_200_SECTORS.values() for t in tickers
]

# Reverse lookup: ticker -> sector
TICKER_SECTOR_MAP = {
    t: sector
    for sector, tickers in NIFTY_200_SECTORS.items()
    for t in tickers
}
