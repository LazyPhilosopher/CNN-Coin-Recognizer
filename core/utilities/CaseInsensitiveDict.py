class CaseInsensitiveDict(dict):
    def __setitem__(self, key, value):
        # Normalize key to lowercase
        super().__setitem__(key.lower(), value)

    def __getitem__(self, key):
        # Retrieve value using normalized lowercase key
        return super().__getitem__(key.lower())

    def __delitem__(self, key):
        # Delete entry using normalized lowercase key
        super().__delitem__(key.lower())

    def get(self, key, default=None):
        # Retrieve value with lowercase key, return default if key not found
        return super().get(key.lower(), default)

    def __contains__(self, key):
        # Check if normalized lowercase key exists
        return super().__contains__(key.lower())

    def update(self, other=None, **kwargs):
        # Update dictionary with lowercase keys
        if other:
            other = {k.lower(): v for k, v in other.items()}
        super().update(other, **kwargs)

    def drop(self, key):
        # Drop method: remove the key if it exists, and return its value or None
        return self.pop(key.lower(), None)