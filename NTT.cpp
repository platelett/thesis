// 速度比不上 SIMD，优点是不需要 pragma
// https://duck.ac/submission/21190

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace internal {
	using u32 = uint32_t;
	using u64 = uint64_t;
	using u128 = __uint128_t;
	const int P = 998244353, SZ = 32;
	u64 size, *w, *iw;

	u64 trans(u64 x) {
		constexpr u64 A = -(u64)P / P + 1;
		constexpr u64 q = ((u128(-(u64)P % P) << 64) + P - 1) / P;
		return x * A + u64((u128)x * q >> 64) + 1;
	}
	u64 mul(u64 a, u64 b) {
		return a * b * (u128)P >> 64;
	}
	u64 add(u64 a, u64 b) {
		if (int64_t((a += b) - P) >= 0) a -= P;
		return a;
	}
	u64 sub(u64 a, u64 b) {
		if (int64_t(a -= b) < 0) a += P;
		return a;
	}
	u64 pow(u64 a, u64 n) {
		u64 b = 1;
		while (n) {
			if (n & 1) b = b * a % P;
			a = a * a % P, n >>= 1;
		}
		return b;
	}
	__attribute((constructor)) void init() {
		size = 1, w = new u64, iw = new u64;
		w[0] = iw[0] = trans(1);
	}
	void extend(u64 n) {
		if (n <= size) return;
		auto _w = new u64[n], _iw = new u64[n];
		memcpy(_w, w, size * 8), w = _w;
		memcpy(_iw, iw, size * 8), iw = _iw;
		while (size < n) {
			u64 wn = pow(3, P / 4 / size), iwn = pow(wn, P - 2);
			for (u64 i = 0; i < size; i++) {
				w[i + size] = trans(mul(w[i], wn));
				iw[i + size] = trans(mul(iw[i], iwn));
			}
			size <<= 1;
		}
	}
	template<int A, int B, int C = 0, class F>
	void repeat(F lambda, u64 i, u64 j, u64 k) {
		if (C == A) return;
		lambda(i, j + C / B, k + C * 2 - C % B);
		repeat<A, B, C + (C < A)>(lambda, i, j, k);
	}
	template<int i, class F> void butterflyA(u64 n, F lambda) {
		if (n <= SZ)
			for (u64 j = 0; j < n / 2 / i; j++)
				for (u64 k = 0; k < i; k++) lambda(i, j, k + 2 * i * j);
		else for (u64 j = 0; j < n / 2 / i; j += SZ / i)
			repeat<SZ, i>(lambda, i, j, i * j * 2);
	}
	template<class F> void butterflyB(u64 i, u64 n, F lambda) {
		for (u64 j = 0; 2 * i * j < n; j++)
			for (u64 k = 0; k < i; k += SZ)
				repeat<SZ, SZ>(lambda, i, j, k + 2 * i * j);
	}
	void DFT(void* A, size_t n) {
		auto a = (u32*)A;
		extend(n);
		auto lambda = [&](u64 i, u64 j, u64 k) {
			u64 x = a[k], y = mul(a[k + i], w[j]);
			a[k] = add(x, y), a[k + i] = sub(x, y);
		};
		for (u64 i = n >> 1; i >= SZ; i >>= 1)
			butterflyB(i, n, lambda);
		butterflyA<16>(n, lambda), butterflyA<8>(n, lambda);
		butterflyA<4>(n, lambda), butterflyA<2>(n, lambda), butterflyA<1>(n, lambda);
	}
	void IDFT(void* A, size_t n) {
		auto a = (u32*)A;
		extend(n);
		auto lambda = [&](u64 i, u64 j, u64 k) {
			u64 x = a[k], y = a[k + i];
			a[k] = add(x, y), a[k + i] = mul(x - y + P, iw[j]);
		};
		butterflyA<1>(n, lambda), butterflyA<2>(n, lambda), butterflyA<4>(n, lambda);
		butterflyA<8>(n, lambda), butterflyA<16>(n, lambda);
		for (u64 i = SZ; i < n; i <<= 1)
			butterflyB(i, n, lambda);
		u64 inv = trans(pow(n, P - 2));
		for (u64 i = 0; i < n; i++) a[i] = mul(a[i], inv);
	}
}

using internal::DFT;
using internal::IDFT;
