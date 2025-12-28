# Acknowledgements

This library is basically me playing with LEGOs - taking brilliant pieces built by others and snapping them together into something (hopefully) useful. Standing on the shoulders of giants, or more accurately, shamelessly copy-pasting from them.

I saw two incredible libraries and thought: "What if these were the same library?" 

Narrator: *They were not meant to be the same library.*

But here we are anyway.

- **[Fastor](https://github.com/romeric/Fastor)** - The SIMD tensor wizardry. I looked at their expression templates and thought "yes, I'll have that, thank you." The contributors clearly sold their souls for those Einstein summation conventions. Worth it.

- **[ensmallen](https://github.com/mlpack/ensmallen)** - The optimization algorithms. Why reinvent Adam when someone already did it perfectly? These folks implemented every optimizer known to humanity, and a few that probably shouldn't exist. I'm borrowing liberally.

- **[SLEEF](https://github.com/shibatch/sleef)** - The SIMD math functions. When you need vectorized sin/cos/exp that's both fast AND accurate, SLEEF is the answer. Their polynomial approximations are chef's kiss.

- **[Sophus](https://github.com/strasdat/Sophus)** - The Lie groups (SO2, SO3, SE2, SE3). The definitive C++ implementation for robotics and computer vision. My lie group implementations are heavily inspired by their elegant API.

- **[datapod](https://github.com/robolibs/datapod)** - The POD foundation. Okay, this one's mine, but it's also standing on the shoulders of other giants, so the chain of "inspiration" continues. It's turtles all the way down.

The plan is simple:
1. Take Fastor's tensor math
2. Take ensmallen's optimizers
3. Add SLEEF's SIMD math functions
4. Add Sophus's Lie groups
5. Glue them together with datapod
6. ???
7. Profit (in knowledge, not money - this is open source)

If this library works, credit goes to them. If it doesn't, that's on me for holding the LEGOs wrong. Or maybe for trying to connect DUPLO to Technic pieces. We'll see.

*No original ideas were harmed in the making of this library. Mostly because there aren't any.*
