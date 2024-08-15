from trading_ig.rest import IGService, ApiExceededException
from trading_ig_config import config
from datetime import datetime
# from tenacity import Retrying, wait_exponential, retry_if_exception_type

# DEFAULT_RETRY = Retrying(
#     wait=wait_exponential(), retry=retry_if_exception_type(ApiExceededException)
# )


def display_top_level_nodes():
    ig_service = get_session()

    response = ig_service.fetch_top_level_navigation_nodes()
    df = response["nodes"]
    for record in df.to_dict("records"):
        print(f"{record['name']} [{record['id']}]")


def display_all_epics():
    ig_service = get_session()
    response = ig_service.fetch_top_level_navigation_nodes()
    df = response["nodes"]
    for record in df.to_dict("records"):
        print(f"{record['name']} [{record['id']}]")
        display_epics_for_node(record["id"], space="  ", ig_service=ig_service)


def display_epics_for_node(node_id=0, space="", ig_service=None):
    if ig_service is None:
        ig_service = get_session()

    sub_nodes = ig_service.fetch_sub_nodes_by_node(node_id)

    if sub_nodes["nodes"].shape[0] != 0:
        rows = sub_nodes["nodes"].to_dict("records")
        for record in rows:
            print(f"{space}{record['name']} [{record['id']}]")
            display_epics_for_node(
                record["id"], space=space + "  ", ig_service=ig_service
            )

    if sub_nodes["markets"].shape[0] != 0:
        cols = sub_nodes["markets"].to_dict("records")
        for record in cols:
            print(
                f"{space}{record['instrumentName']} ({record['expiry']}): "
                f"{record['epic']}"
            )


def get_session():
    ig_service = IGService(
        config.username,
        config.password,
        config.api_key,
        config.acc_type,
        acc_number=config.acc_number
    )
    ig_service.create_session(version="3")
    return ig_service


if __name__ == "__main__":
    # display_top_level_nodes()

    # display_all_epics()
    ig_service = get_session()
    # accounts = ig_service.fetch_accounts()
    # print("accounts:\n%s" % accounts)
    # ig_service.create_open_position()
    # from_date = datetime(2021, 1, 1)
    # activities = ig_service.fetch_account_activity(from_date=from_date)
    # activities
    ig_service.create_open_position(
        currency_code='GBP',
        direction='BUY',
        epic='CS.D.USCGC.TODAY.IP',
        order_type='MARKET',
        expiry='DFB',
        force_open='false',
        guaranteed_stop='false',
        size=0.01, level=None,
        limit_distance=None,
        limit_level=None,
        quote_id=None,
        stop_level=None,
        stop_distance=None,
        trailing_stop=None,
        trailing_stop_increment=None)
