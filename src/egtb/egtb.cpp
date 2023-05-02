#include "egtb.h"

#include "util/algo.h"

Piece_Group::Piece_Group(const std::vector<Piece>& pcs) :
	m_num_pieces(pcs.size()),
	m_pieces{},
	m_table_size(0),
	m_compress_size(0),
	m_placements{},
	m_weights{},
	m_unique_placement_indices{},
	m_opp_piece_group(nullptr)
{
	if (m_num_pieces >= MAX_PIECE_GROUP_SIZE)
		throw std::runtime_error("Too many pieces in piece group.");

	if (m_num_pieces == 0)
		throw std::runtime_error("Trying to form an empty piece group.");

	std::memcpy(m_pieces, pcs.data(), m_num_pieces);

	size_t num_raw_placements = 1;
	for (size_t i = 0; i < m_num_pieces; ++i)
	{
		m_weights[i] = num_raw_placements;
		num_raw_placements *= possible_sq_nb(m_pieces[i]);
	}

	m_unique_placement_indices.resize(num_raw_placements);
	m_unique_to_non_unique.resize(num_raw_placements);

	// Helpers to form a unique key for each configuration
	// of pieces and squares (placements).
	// We need this for fast and safe identification of duplicates
	// and reflections.
	using SafePositionKey = std::array<uint16_t, MAX_PIECE_GROUP_SIZE>;
	auto make_keys = [&](const Placement& placement) {
		SafePositionKey base_key;
		SafePositionKey mirr_key;
		base_key.fill(0);
		mirr_key.fill(0);

		for (size_t i = 0; i < m_num_pieces; ++i)
		{
			const Square sq = placement[i];
			const Piece pc = m_pieces[i];
			mirr_key[i] = sq_file_mirror(sq) * PIECE_NB + pc;
			base_key[i] = sq * PIECE_NB + pc;
		}

		std::sort(base_key.begin(), base_key.end());
		std::sort(mirr_key.begin(), mirr_key.end());
		return std::make_pair(base_key, mirr_key);
	};

	std::set<SafePositionKey> key_set;
	std::vector<std::pair<Piece_Group::Placement, bool>> sq_vec;

	std::vector<size_t> radices;
	for (size_t i = 0; i < m_num_pieces; ++i)
		radices.emplace_back(possible_sq_nb(m_pieces[i]));

	// Go through all possible square configurations.
	// The iteration order is important. The least significant square
	// is always changed first.
	for (const auto& ixs : Mixed_Radix(Const_Span(radices)))
	{
		Piece_Group::Placement tmp_list;
		for (size_t i = 0; i < m_num_pieces; ++i)
			tmp_list.add(possible_sq(m_pieces[i], ixs[i]));

		// We are only interested in placements that are legal.
		if (!tmp_list.are_all_squares_unique())
			continue;

		// We compute two keys for each placement.
		// One is corresponding to it directly, the other
		// to the left-right mirror of it.
		const auto [key, mirror] = make_keys(tmp_list);

		// If we have not seen this placement yet.
		if (key_set.find(key) == key_set.end())
		{
			// We find out if the left-right mirror of this placement has been seen already.
			const bool is_mirr = key_set.find(mirror) != key_set.end();
			key_set.insert(key);
			sq_vec.emplace_back(tmp_list, is_mirr); // and we note that fact for sorting later.

			// When using a 16-bit index not all Piece_Groups are representable.
			// We can't proceed in these cases.
			if (sq_vec.size() > Placement_Index::MAX_INDEX)
				throw std::runtime_error("PieceGroup too big.");
		}
	}

	m_table_size = sq_vec.size();
	ASSERT(m_table_size <= Placement_Index::MAX_INDEX);

	m_placements.reserve(m_table_size);
	m_mirr_placement_index.resize(m_table_size);

	// We partition the placements. We put placements that have had 
	// their left-right mirrors already present in the list later in the list.
	for (const auto& [sqvec, is_mirr] : sq_vec)
		if (!is_mirr)
			m_placements.emplace_back(sqvec);
	m_compress_size = m_placements.size();
	for (const auto& [sqvec, is_mirr] : sq_vec)
		if (is_mirr)
			m_placements.emplace_back(sqvec);
	ASSERT(m_placements.size() == m_table_size);

	// Inspects the pieces in this group and produces ranges of equal pieces.
	// These are used for resolving identicaly configurations, as
	// changing the order of two identical pieces on the board doesn't matters,
	// so the pieces in these ranges can be permuted without changing the placement's meaning.
	auto make_permutation_ranges = [&]() {
		std::vector<std::pair<size_t, size_t>> permutation_ranges;
		for (size_t begin = 0, end = 0; begin < m_num_pieces; begin = end)
		{
			while (end < m_num_pieces && m_pieces[begin] == m_pieces[end])
				++end;
			if (end - begin >= 2)
				permutation_ranges.emplace_back(begin, end);
		}
		return permutation_ranges;
	};

	// Fills the lookup table m_unique_placement_indices, accounting for all
	// possible placements that lead to the same unique index.
	auto map_to_unique_index = [this, permutation_ranges = make_permutation_ranges()](Piece_Group::Placement tmp_list, Full_Placement_Index i) {
		// We can permute each piece within each same-piece range without changing the placement's meaning.
		Multi_Permuter<Piece_Group::Placement> permuter(tmp_list, permutation_ranges);
		do
			m_unique_placement_indices[non_unique_placement_index(tmp_list)] = i;
		while (permuter.try_advance());
	};

	// First fill the base index.
	for (Placement_Index i = ZERO_INDEX; i < m_table_size; ++i)
	{
		ASSERT(m_placements[i].size() == m_num_pieces);
		ASSERT(m_unique_placement_indices[non_unique_placement_index(m_placements[i])].base() == ZERO_INDEX);
		ASSERT(m_unique_placement_indices[non_unique_placement_index(m_placements[i])].mirr() == ZERO_INDEX);

		map_to_unique_index(m_placements[i], Full_Placement_Index(i));
		m_unique_to_non_unique[i] = static_cast<uint32_t>(non_unique_placement_index(m_placements[i]));
	}

	// Then fill the left-right mirrored index based on the base indices.
	// We need the base indices fully populated for this.
	for (Placement_Index i = ZERO_INDEX; i < m_table_size; ++i)
	{
		ASSERT(m_placements[i].size() == m_num_pieces);
		ASSERT((m_unique_placement_indices[non_unique_placement_index(m_placements[i])].base()) == i);

		const Placement_Index mir_idx = m_unique_placement_indices[non_unique_placement_index(m_placements[i].with_mirrored_files())].base();

		map_to_unique_index(m_placements[i], Full_Placement_Index(i, mir_idx));

		// Also fill the lookup table for mirrored indices from base indices.
		m_mirr_placement_index[i] = mir_idx;
	}

	// Fill the lookup table for raw index differences on quiet move.
	for (Placement_Index i = ZERO_INDEX; i < m_table_size; ++i)
	{
		const Placement& base_list = m_placements[i];

		for (size_t j = 0; j < m_num_pieces; ++j)
		{
			const Piece p = m_pieces[j];
			const Square from = base_list[j];
			for (Square to = SQ_START; to < SQ_END; ++to)
			{
				// We only care about squares that the piece can be on.
				if (possible_sq_index(p, to) == -1)
					continue;

				Placement moved_list = base_list;
				moved_list[j] = to;
				m_diff_on_move[j][from][to] = static_cast<int32_t>(non_unique_placement_index(moved_list))
										    - static_cast<int32_t>(non_unique_placement_index(base_list));
			}
		}
	}
}
