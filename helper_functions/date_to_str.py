def date_to_str(date):
    date = str(date).zfill(4)
    date = f'20{date[:2]}-20{date[2:]}'

    return date