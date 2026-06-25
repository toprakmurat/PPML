import time
from typing import Optional, Union

import numpy as np

from concrete import fhe

class StaticKeyValueDatabase:
	number_of_entries: int
	key_size: int
	value_size: int
	chunk_size: int

	_number_of_key_chunks: int
	_number_of_value_chunks: int
	_state_shape: tuple[int, ...]

	module: fhe.Module
	state: Optional[fhe.Value]

	def __init__(
		self,
		number_of_entries: int,
		key_size: int = 32,
		value_size: int = 32,
		chunk_size: int = 4,
		compiled: bool = True,
		configuration: Optional[fhe.Configuration] = None,
	):
		self.number_of_entries = number_of_entries
		self.key_size = key_size
		self.value_size = value_size
		self.chunk_size = chunk_size
		self._number_of_key_chunks = key_size // chunk_size
		self._number_of_value_chunks = value_size // chunk_size
		self._state_shape = (
			number_of_entries,
			1 + self._number_of_key_chunks + self._number_of_value_chunks,
		)

		if compiled:
			if configuration is None:
				configuration = fhe.Configuration()

			self.module = self._module(
				configuration.fork(
					fhe.MultivariateStrategy.PROMOTED
				)
			)

		self.state = None

	def _encode(self, number: int, width: int) -> np.ndarray:
		pass

	def _decode(self, encoded_number: np.ndarray) -> int:
		pass

	def encode_key(self, key: int) -> np.ndarray:
		return self._encode(key, width=self.key_size)

	def decode_key(self, encoded_key: np.ndarray) -> int:
		return self._decode(encoded_key)

	def encode_value(self, value: int) -> np.ndarray:
		return self._encode(value, width=self.value_size)

	def decode_value(self, encoded_value: np.ndarray) -> int:
		return self._decode(encoded_value)

	def _module(self, configuration: fhe.Configuration) -> fhe.Module:
		flag_slice = 0
		key_slice = slice(1, 1 + self._number_of_key_chunks)
		value_slice = slice(1, 1 + self._number_of_value_chunks)

		chunk_size = self.chunk_size
		number_of_entries = self.number_of_entries
		number_of_key_chunks = self._number_of_key_chunks
		state_shape = self._state_shape

		@fhe.module()
		class StaticKeyValueDatabaseModule:
			@fhe.function({"state": "clear"})
			def reset(state):
				return state + fhe.zero();

			@fhe.function({"state": "encrypted", "key": "encrypted", "value": "encrypted"})
			def insert(state, key, value):
				flags = state[:, flag_slice]

				selection = fhe.zeros(number_of_entries)

				found = fhe.zero()
				for i in range(number_of_entries):
					is_selected = fhe.multivariate(
						lambda found, flag: int(found == 0 and flag == 0)
					)(found, flags[i])

					selection[i] = is_selected
					found += is_selected

				state_update = fhe.zeros(state_shape)
				state_update[:, flag_slice] = selection

				selection = selection.reshape((-1, 1))

                key_update = fhe.multivariate(lambda selection, key: selection * key)(
                    selection, key
                )
                value_update = fhe.multivariate(lambda selection, value: selection * value)(
                    selection, value
                )

                state_update[:, key_slice] = key_update
                state_update[:, value_slice] = value_update

                new_state = state + state_update
                return fhe.refresh(new_state)
