use std::cmp::min;

pub struct AlignedIter<'a> {
    data: &'a [u8],
    align_bytes: usize,
    offset: usize,
    processed: usize,
}

impl<'a> AlignedIter<'a> {
    #[allow(unused)]
    pub fn new(data: &'a [u8], align_bytes: usize) -> AlignedIter<'a> {
        AlignedIter {
            data,
            align_bytes,
            offset: data.as_ptr().align_offset(16),
            processed: 0,
        }
    }
}

impl<'a> Iterator for AlignedIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.processed == self.data.len() {
            None
        } else {
            let len = min(
                self.data.len() - self.processed,
                if self.processed < self.offset {
                    self.offset
                } else {
                    self.align_bytes
                },
            );
            let start = self.processed;
            self.processed += len;
            Some(&self.data[start..start + len])
        }
    }
}
