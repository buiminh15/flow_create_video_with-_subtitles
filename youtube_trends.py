"""Xem video trending theo khu vực bằng YouTube Data API v3.

Cần API key (khác key Gemini) lấy tại https://console.cloud.google.com/apis/credentials sau khi
Enable "YouTube Data API v3" cho project. Đặt vào .env: YOUTUBE_API_KEY=...
"""
import os

import requests

BASE = "https://www.googleapis.com/youtube/v3"


def _key(api_key):
    key = (api_key or "").strip() or os.environ.get("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError("Chưa có YouTube API key. Lấy tại console.cloud.google.com/apis/credentials")
    return key


def _get(path, api_key, **params):
    params["key"] = _key(api_key)
    r = requests.get(f"{BASE}/{path}", params=params, timeout=30)
    data = r.json()
    if "error" in data:
        raise RuntimeError(data["error"].get("message", str(data["error"])))
    return data


def video_categories(api_key, region="TH"):
    """Trả về dict {id: tên danh mục} áp dụng cho khu vực (ví dụ: Music, Gaming, Entertainment)."""
    data = _get("videoCategories", api_key, part="snippet", regionCode=region)
    return {item["id"]: item["snippet"]["title"] for item in data.get("items", [])}


def trending(api_key, region="TH", category_id="", max_results=25):
    """Top video trending tại khu vực. Trả về list dict: id, title, channel, views, likes, comments,
    published, tags, url, thumbnail, category_id."""
    params = {"part": "snippet,statistics", "chart": "mostPopular", "regionCode": region,
              "maxResults": min(50, max_results)}
    if category_id:
        params["videoCategoryId"] = category_id
    data = _get("videos", api_key, **params)
    out = []
    for item in data.get("items", []):
        sn, st = item["snippet"], item.get("statistics", {})
        out.append({
            "id": item["id"],
            "title": sn.get("title", ""),
            "channel": sn.get("channelTitle", ""),
            "views": int(st.get("viewCount", 0)),
            "likes": int(st.get("likeCount", 0)) if "likeCount" in st else 0,
            "comments": int(st.get("commentCount", 0)) if "commentCount" in st else 0,
            "published": sn.get("publishedAt", "")[:10],
            "tags": sn.get("tags", []),
            "category_id": sn.get("categoryId", ""),
            "thumbnail": (sn.get("thumbnails", {}).get("medium") or {}).get("url", ""),
            "url": f"https://www.youtube.com/watch?v={item['id']}",
        })
    return out


def search(api_key, query, region="TH", max_results=25, order="viewCount"):
    """Tìm video theo từ khoá, sắp theo order (viewCount|date|rating|relevance). Dùng khi cần lọc
    theo chủ đề cụ thể thay vì chỉ xem top trending chung."""
    data = _get("search", api_key, part="snippet", q=query, type="video", regionCode=region,
                maxResults=min(50, max_results), order=order)
    ids = [item["id"]["videoId"] for item in data.get("items", [])]
    if not ids:
        return []
    stats = _get("videos", api_key, part="statistics", id=",".join(ids))
    stat_by_id = {v["id"]: v.get("statistics", {}) for v in stats.get("items", [])}
    out = []
    for item in data.get("items", []):
        vid, sn = item["id"]["videoId"], item["snippet"]
        st = stat_by_id.get(vid, {})
        out.append({
            "id": vid,
            "title": sn.get("title", ""),
            "channel": sn.get("channelTitle", ""),
            "views": int(st.get("viewCount", 0)) if "viewCount" in st else 0,
            "likes": int(st.get("likeCount", 0)) if "likeCount" in st else 0,
            "comments": int(st.get("commentCount", 0)) if "commentCount" in st else 0,
            "published": sn.get("publishedAt", "")[:10],
            "tags": [],
            "category_id": "",
            "thumbnail": (sn.get("thumbnails", {}).get("medium") or {}).get("url", ""),
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    return out
