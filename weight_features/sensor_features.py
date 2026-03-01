import os
import pandas as pd
import numpy as np
from datetime import datetime
import math

DATA_DIR = "dataset"
OUTPUT_FILE = "sensor_daily_features.csv"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0 # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def process_phonelock(df):
    if df is None or df.empty: return pd.DataFrame()
    df.columns = df.columns.str.strip()
    if 'start' not in df.columns or 'end' not in df.columns: return pd.DataFrame()
    df['start'] = pd.to_numeric(df['start'], errors='coerce')
    df['end'] = pd.to_numeric(df['end'], errors='coerce')
    df = df.dropna(subset=['start', 'end'])
    df['start_dt'] = pd.to_datetime(df['start'], unit='s', errors='coerce')
    df['end_dt'] = pd.to_datetime(df['end'], unit='s', errors='coerce')
    df = df.dropna(subset=['start_dt', 'end_dt'])
    df['date'] = df['start_dt'].dt.date
    df['duration'] = df['end'] - df['start']
    df['is_night'] = (df['start_dt'].dt.hour >= 0) & (df['start_dt'].dt.hour < 6)
    
    daily = df.groupby('date').agg(
        unlock_count=('start', 'count'),
        avg_session_sec=('duration', 'mean'),
        total_unlocked_sec=('duration', 'sum'),
        night_unlocks=('is_night', 'sum')
    ).reset_index()
    return daily

def process_dark(df):
    if df is None or df.empty: return pd.DataFrame()
    df.columns = df.columns.str.strip()
    if 'start' not in df.columns or 'end' not in df.columns: return pd.DataFrame()
    df['start'] = pd.to_numeric(df['start'], errors='coerce')
    df['end'] = pd.to_numeric(df['end'], errors='coerce')
    df = df.dropna(subset=['start', 'end'])
    df['start_dt'] = pd.to_datetime(df['start'], unit='s', errors='coerce')
    df['end_dt'] = pd.to_datetime(df['end'], unit='s', errors='coerce')
    df = df.dropna(subset=['start_dt', 'end_dt'])
    df['date'] = df['start_dt'].dt.date
    df['duration_hrs'] = (df['end'] - df['start']) / 3600.0
    df['is_night'] = (df['start_dt'].dt.hour >= 0) & (df['start_dt'].dt.hour < 6)
    
    daily = df.groupby('date').agg(
        total_dark_hrs=('duration_hrs', 'sum'),
        dark_fragments=('start', 'count'),
        longest_dark_streak_hrs=('duration_hrs', 'max'),
    ).reset_index()
    
    night_df = df[df['is_night']]
    night_daily = night_df.groupby('date')['duration_hrs'].sum().reset_index().rename(columns={'duration_hrs': 'night_dark_hrs'})
    daily = pd.merge(daily, night_daily, on='date', how='left')
    daily['night_dark_hrs'] = daily['night_dark_hrs'].fillna(0)
    return daily

def process_gps(df):
    if df is None or df.empty: return pd.DataFrame()
    df.columns = df.columns.str.strip()
    if 'time' not in df.columns or 'latitude' not in df.columns or 'longitude' not in df.columns: return pd.DataFrame()
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df = df.dropna(subset=['time', 'latitude', 'longitude'])
    df = df.sort_values('time')
    df['dt'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
    df = df.dropna(subset=['dt'])
    df['date'] = df['dt'].dt.date
    
    df['lat_shift'] = df['latitude'].shift(1)
    df['lon_shift'] = df['longitude'].shift(1)
    
    def calc_dist(row):
        if pd.isna(row['lat_shift']): return 0
        return haversine(row['latitude'], row['longitude'], row['lat_shift'], row['lon_shift'])
        
    df['distance'] = df.apply(calc_dist, axis=1)
    
    df['lat_round'] = df['latitude'].round(3)
    df['lon_round'] = df['longitude'].round(3)
    df['loc_id'] = df['lat_round'].astype(str) + "_" + df['lon_round'].astype(str)
    
    def loc_entropy(x):
        counts = x.value_counts(normalize=True)
        return -np.sum(counts * np.log(counts + 1e-9))
        
    def loc_top_pct(x):
        counts = x.value_counts(normalize=True)
        return counts.iloc[0] if not counts.empty else 0
        
    if df.empty: return pd.DataFrame()
    daily = df.groupby('date').agg(
        total_distance_km=('distance', 'sum'),
        num_clusters=('loc_id', 'nunique'),
        location_entropy=('loc_id', loc_entropy),
        time_at_top_location_pct=('loc_id', loc_top_pct)
    ).reset_index()
    return daily

def process_conversation(df):
    if df is None or df.empty: return pd.DataFrame()
    df.columns = df.columns.str.strip()
    if 'start_timestamp' not in df.columns or 'end_timestamp' not in df.columns: return pd.DataFrame()
    df['start_timestamp'] = pd.to_numeric(df['start_timestamp'], errors='coerce')
    df['end_timestamp'] = pd.to_numeric(df['end_timestamp'], errors='coerce')
    df = df.dropna(subset=['start_timestamp', 'end_timestamp'])
    df['start_dt'] = pd.to_datetime(df['start_timestamp'], unit='s', errors='coerce')
    df = df.dropna(subset=['start_dt'])
    df['date'] = df['start_dt'].dt.date
    df['duration_min'] = (df['end_timestamp'] - df['start_timestamp']) / 60.0
    
    daily = df.groupby('date').agg(
        convo_count=('start_timestamp', 'count'),
        total_convo_min=('duration_min', 'sum'),
        avg_convo_length_min=('duration_min', 'mean')
    ).reset_index()
    return daily

def process_call_log(df):
    if df is None or df.empty: return pd.DataFrame()
    df.columns = df.columns.str.strip()
    if 'timestamp' not in df.columns or 'CALLS_type' not in df.columns or 'CALLS_duration' not in df.columns: return pd.DataFrame()
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    df = df.dropna(subset=['dt'])
    df['date'] = df['dt'].dt.date
    
    df['incoming'] = df['CALLS_type'] == 1
    df['outgoing'] = df['CALLS_type'] == 2
    df['missed'] = df['CALLS_type'] == 3
    
    daily = df.groupby('date').agg(
        incoming_calls=('incoming', 'sum'),
        outgoing_calls=('outgoing', 'sum'),
        missed_calls=('missed', 'sum'),
        total_call_min=('CALLS_duration', lambda x: x.sum() / 60.0)
    ).reset_index()
    return daily

def process_sms(df):
    if df is None or df.empty: return pd.DataFrame()
    df.columns = df.columns.str.strip()
    if 'timestamp' not in df.columns: return pd.DataFrame()
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
    df = df.dropna(subset=['dt'])
    df['date'] = df['dt'].dt.date
    
    df['received'] = df['MESSAGE_type'] == 1 if 'MESSAGE_type' in df.columns else False
    if 'MESSAGES_type' in df.columns:
         df['received'] = df['MESSAGES_type'] == 1
    df['sent'] = df['MESSAGE_type'] == 2 if 'MESSAGE_type' in df.columns else False
    if 'MESSAGES_type' in df.columns:
         df['sent'] = df['MESSAGES_type'] == 2
    
    daily = df.groupby('date').agg(
        sms_sent=('sent', 'sum'),
        sms_received=('received', 'sum')
    ).reset_index()
    return daily

def process_phonecharge(df):
    if df is None or df.empty: return pd.DataFrame()
    df.columns = df.columns.str.strip()
    if 'start' not in df.columns or 'end' not in df.columns: return pd.DataFrame()
    df['start'] = pd.to_numeric(df['start'], errors='coerce')
    df['end'] = pd.to_numeric(df['end'], errors='coerce')
    df = df.dropna(subset=['start', 'end'])
    df['start_dt'] = pd.to_datetime(df['start'], unit='s', errors='coerce')
    df = df.dropna(subset=['start_dt'])
    df['date'] = df['start_dt'].dt.date
    df['duration_hrs'] = (df['end'] - df['start']) / 3600.0
    df['is_night'] = (df['start_dt'].dt.hour >= 0) & (df['start_dt'].dt.hour < 6)
    
    daily = df.groupby('date').agg(
        charge_sessions=('start', 'count')
    ).reset_index()
    
    night_df = df[df['is_night']]
    night_daily = night_df.groupby('date')['duration_hrs'].sum().reset_index().rename(columns={'duration_hrs': 'night_charge_hrs'})
    
    daily = pd.merge(daily, night_daily, on='date', how='left')
    daily['night_charge_hrs'] = daily['night_charge_hrs'].fillna(0)
    return daily

def process_wifi_location(df):
    if df is None or df.empty: return pd.DataFrame()
    df.columns = df.columns.str.strip()
    if 'time' not in df.columns or 'location' not in df.columns: return pd.DataFrame()
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    df['dt'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
    df = df.dropna(subset=['dt'])
    df['date'] = df['dt'].dt.date
    
    def top_pct(x):
        counts = x.value_counts(normalize=True)
        return counts.iloc[0] if not counts.empty else 0
        
    daily = df.groupby('date').agg(
        unique_locations=('location', 'nunique'),
        top_location_pct=('location', top_pct)
    ).reset_index()
    return daily

def process_bluetooth(df):
    if df is None or df.empty: return pd.DataFrame()
    df.columns = df.columns.str.strip()
    if 'time' not in df.columns or 'MAC' not in df.columns: return pd.DataFrame()
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    df['dt'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
    df = df.dropna(subset=['dt'])
    df['date'] = df['dt'].dt.date
    
    aggs = df.groupby(['date', 'time']).agg(nearby_devices=('MAC', 'count')).reset_index()
    daily = aggs.groupby('date').agg(avg_nearby_devices=('nearby_devices', 'mean')).reset_index()
    
    return daily

def safe_read(file_path):
    if not os.path.exists(file_path): return None
    try:
        df = pd.read_csv(file_path)
        return df
    except:
        return None

def main():
    print("Starting sensor feature engineering...")
    all_users = [f"u{str(i).zfill(2)}" for i in range(60)]
    
    all_daily_features = []
    
    for user in all_users:
        print(f"Processing user {user}...")
        
        # Load and process each file if exists
        phonelock = process_phonelock(safe_read(f"{DATA_DIR}/sensing/phonelock/phonelock_{user}.csv"))
        dark = process_dark(safe_read(f"{DATA_DIR}/sensing/dark/dark_{user}.csv"))
        gps = process_gps(safe_read(f"{DATA_DIR}/sensing/gps/gps_{user}.csv"))
        conv = process_conversation(safe_read(f"{DATA_DIR}/sensing/conversation/conversation_{user}.csv"))
        call = process_call_log(safe_read(f"{DATA_DIR}/call_log/call_log_{user}.csv"))
        sms = process_sms(safe_read(f"{DATA_DIR}/sms/sms_{user}.csv"))
        charge = process_phonecharge(safe_read(f"{DATA_DIR}/sensing/phonecharge/phonecharge_{user}.csv"))
        wifi = process_wifi_location(safe_read(f"{DATA_DIR}/sensing/wifi_location/wifi_location_{user}.csv"))
        bt = process_bluetooth(safe_read(f"{DATA_DIR}/sensing/bluetooth/bt_{user}.csv"))
        
        # Combine all features for the user on 'date'
        dfs = [d for d in [phonelock, dark, gps, conv, call, sms, charge, wifi, bt] if not d.empty]
        if not dfs: continue
        
        user_daily = dfs[0]
        for d in dfs[1:]:
            user_daily = pd.merge(user_daily, d, on='date', how='outer')
            
        user_daily.insert(0, 'uid', user)
        # We need to fill NaNs based on specific logic.
        # Drop days with >50% missing features (excluding uid and date)
        feature_cols = [c for c in user_daily.columns if c not in ['uid', 'date']]
        
        threshold = len(feature_cols) / 2.0
        user_daily = user_daily.dropna(thresh=int(threshold), subset=feature_cols)
        
        # Fill remaining NaNs with the student's median
        user_daily[feature_cols] = user_daily[feature_cols].fillna(user_daily[feature_cols].median())
        
        all_daily_features.append(user_daily)
        
    if all_daily_features:
        final_df = pd.concat(all_daily_features, ignore_index=True)
        # Re-sort
        final_df = final_df.sort_values(by=['uid', 'date'])
        final_df = final_df.fillna(0)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Finished! Output written to {OUTPUT_FILE} with shape {final_df.shape}")
    else:
        print("No feature data extracted.")

if __name__ == "__main__":
    main()
