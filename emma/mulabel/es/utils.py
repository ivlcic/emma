import logging
from datetime import timedelta, datetime
from typing import Callable, Dict, Any, List

from elasticsearch import Elasticsearch
from tqdm import tqdm

logger = logging.getLogger('mulabel.es.utils')


def _get_date_ranges(start_date: datetime, end_date: datetime, interval_hours: int = 24):
    """Generate date ranges for chunked processing"""
    current_date = start_date
    while current_date < end_date:
        next_date = min(current_date + timedelta(hours=interval_hours), end_date)
        yield current_date, next_date
        current_date = next_date


def load_data(client: Elasticsearch, collection: str, start_date: datetime, end_date: datetime,
              callback: Callable[[Dict[str, Any]], bool]) -> None:
    # Process data in date chunks
    date_ranges = list(_get_date_ranges(start_date, end_date))

    for start, end in tqdm(date_ranges, desc="Processing date ranges"):
        # Get documents for the current date range
        query = {
            "size": 10000,
            "query": {
                "range": {
                    "date": {
                        "gte": start.astimezone().isoformat(),
                        "lt": end.astimezone().isoformat()
                    }
                }
            },
            "_source": ["a_uuid", "a_id", "date", "text", "m_bge_m3"],
            "sort": {
                "date": "asc"
            }
        }

        response = client.search(
            index=collection,
            body=query,
        )
        # logger.info(
        #     'Response for range [%s][%s]', start.astimezone().isoformat(), end.astimezone().isoformat()
        # )
        hits = response['hits']['hits']
        for doc in hits:
            if not callback(doc['_source']):
                break


def find_similar(client: Elasticsearch, collection: str, uuid: str, vector: List[float],
                 callback: Callable[[Dict[str, Any], float], bool]) -> None:
    query = {
        "size": 50,
        "query": {
            "bool": {
                "must_not": [
                    {
                        "term": {
                            "a_uuid": uuid
                        }
                    }
                ],
                "must": {
                    "knn": {
                        "field": "m_bge_m3",
                        "query_vector": vector,
                        "k": 50
                    }
                },
            }
        },
        "_source": ["a_uuid", "a_id", "date", "text", "m_bge_m3"],
        "sort": {
            "_score": "desc"
        }
    }

    response = client.search(index=collection, body=query)
    # logger.info(
    #     'Response for range [%s][%s]', start.astimezone().isoformat(), end.astimezone().isoformat()
    # )
    for doc in response['hits']['hits']:
        if not callback(doc['_source'], doc['_score']):
            break
