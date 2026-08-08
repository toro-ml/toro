type ClassName = string | false | null | undefined;

export function classNames(...names: ClassName[]) {
  return names.filter((name): name is string => Boolean(name)).join(" ");
}
